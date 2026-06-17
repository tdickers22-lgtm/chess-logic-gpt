"""Verifiable rewards for RLVR (GRPO/RLOO) and honest evaluation.

Every trainable domain here is *programmatically checkable*, so the same reward
interface drives both eval and RL:

    score(record, model_output) -> RewardResult   # .score in [0, 1]

- chess (tactics puzzles): parse the model's move(s) from <answer> and replay
  them against the puzzle's forced solution line using python-chess. First move
  correct earns partial credit; the whole line earns full credit. Illegal or
  wrong first move earns zero. No engine required -- the Lichess solution line
  is ground truth.
- memory: exact match of the model's <answer> against the known solution stored
  in record metadata (facts table, rotation, or grid solution).

The model is trained to emit a shared trace skeleton:

    <reasoning> ...work, name the motif, calculate the line... </reasoning>
    <answer> final move(s) / answer </answer>

so a single parser serves all domains.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class RewardResult:
    score: float
    correct: bool
    detail: str


def extract_tag(text: str, tag: str) -> str | None:
    """Return the inner text of the last <tag>...</tag> block, or None."""
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def final_answer(text: str) -> str:
    """The <answer> block if present, else the whole (stripped) text."""
    answer = extract_tag(text, "answer")
    return answer if answer is not None else text.strip()


# --------------------------------------------------------------------------- #
# Chess tactics
# --------------------------------------------------------------------------- #

def _parse_move(board: chess.Board, token: str) -> chess.Move | None:
    """Parse one token as a legal move on `board` (UCI first, then SAN)."""
    token = token.strip().rstrip(".,;:!)?")
    if not token:
        return None
    try:
        move = chess.Move.from_uci(token.lower())
        if move in board.legal_moves:
            return move
    except ValueError:
        pass
    try:
        return board.parse_san(token)
    except ValueError:
        return None


def parse_move_line(fen: str, text: str) -> list[chess.Move]:
    """Extract an ordered list of legal moves from free text, in board context.

    Non-move tokens (prose, move numbers like ``1.``) are skipped so a line such
    as ``1. Rd8+ Kxd8 2. Qe8#`` parses cleanly.
    """
    board = chess.Board(fen)
    moves: list[chess.Move] = []
    for raw in re.split(r"\s+", text.strip()):
        tok = re.sub(r"^\d+\.+", "", raw.strip())
        if not tok:
            continue
        move = _parse_move(board, tok)
        if move is None:
            continue
        moves.append(move)
        board.push(move)
    return moves


def _tokenize_moves(text: str) -> list[str]:
    """Whitespace tokens with leading move numbers (``1.``) stripped."""
    return [re.sub(r"^\d+\.+", "", tok) for tok in re.split(r"\s+", text.strip())]


def _next_parsed_move(tokens: list[str], start: int, board: chess.Board) -> tuple[chess.Move | None, int]:
    """First token at/after `start` that is a legal move on `board`; skip prose."""
    i = start
    while i < len(tokens):
        move = _parse_move(board, tokens[i])
        i += 1
        if move is not None:
            return move, i
    return None, i


def _match_solver_only(fen: str, line: list[str], tokens: list[str]) -> int:
    """Model supplies only the solver's moves; we auto-play the forced replies."""
    board = chess.Board(fen)
    ti = 0
    matched = 0
    for i, exp_uci in enumerate(line):
        if i % 2 == 0:  # solver ply -- from the model
            move, ti = _next_parsed_move(tokens, ti, board)
            if move is None or move.uci() != exp_uci:
                break
            board.push(move)
            matched += 1
        else:  # opponent ply -- forced reply from the known line
            try:
                reply = chess.Move.from_uci(exp_uci)
            except ValueError:
                break
            if reply not in board.legal_moves:
                break
            board.push(reply)
    return matched


def _match_full_line(fen: str, line: list[str], tokens: list[str]) -> int:
    """Model spells out the whole line (solver + replies); count solver moves covered."""
    board = chess.Board(fen)
    ti = 0
    plies = 0
    for exp_uci in line:
        move, ti = _next_parsed_move(tokens, ti, board)
        if move is None or move.uci() != exp_uci:
            break
        board.push(move)
        plies += 1
    return (plies + 1) // 2


def _match_solver_moves(fen: str, line: list[str], text: str) -> int:
    """Best of both answer conventions; never exceeds the number of solver moves.

    We can't reliably tell whether the model wrote only its own moves or the
    full line by peeking (consecutive plies often target the same square, e.g.
    Rd8 / ...Rxd8 / Rxd8#), so we score both interpretations and take the max.
    """
    tokens = _tokenize_moves(text)
    return max(
        _match_solver_only(fen, line, tokens),
        _match_full_line(fen, line, tokens),
    )


def score_chess_puzzle(record: dict, output: str) -> RewardResult:
    """Reward = fraction of the solver's moves matched, from the puzzle FEN.

    metadata.fen        position with the solver to move
    metadata.line_uci   forced continuation as alternating UCI moves
                        [solver, reply, solver, reply, ...]
    """
    md = record.get("metadata", {})
    fen = md.get("fen")
    line = md.get("line_uci") or md.get("solution_uci")
    if not fen or not line:
        return RewardResult(0.0, False, "missing fen/line_uci in metadata")

    n = len(line[0::2])
    if n == 0:
        return RewardResult(0.0, False, "empty solution")

    matched = _match_solver_moves(fen, line, final_answer(output))
    return RewardResult(matched / n, matched == n, f"matched {matched}/{n} solver move(s)")


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #

def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _contains(haystack: str, needle: str) -> bool:
    return f" {_norm(needle).strip()} " in f" {' '.join(_norm(haystack).split())} "


def score_memory(record: dict, output: str) -> RewardResult:
    task = record.get("task")
    md = record.get("metadata", {})
    answer = final_answer(output)

    if task == "working_memory_fact_recall":
        target = next((f for f in md.get("facts", []) if f.startswith(md.get("query", "\0"))), None)
        if not target:
            return RewardResult(0.0, False, "no target fact in metadata")
        m = re.search(r"has the (.+?)\.", target)
        needed = m.group(1) if m else ""
        ok = _contains(answer, needed)
        return RewardResult(1.0 if ok else 0.0, ok, f"need '{needed}'")

    if task == "working_memory_multi_query":
        needed = md.get("needed", [])
        if not needed:
            return RewardResult(0.0, False, "no needed answers in metadata")
        hits = sum(1 for phrase in needed if _contains(answer, phrase))
        frac = hits / len(needed)
        return RewardResult(frac, frac == 1.0, f"{hits}/{len(needed)} queries answered")

    if task == "working_memory_order_transform":
        transformed = md.get("transformed", [])
        if len(transformed) >= 5:
            ok = _contains(answer, transformed[1]) and _contains(answer, transformed[4])
            return RewardResult(1.0 if ok else 0.0, ok, f"need {transformed[1]}, {transformed[4]}")
        return RewardResult(0.0, False, "transformed list too short")

    if task == "constraint_reasoning_grid":
        solution = md.get("solution", {})
        if solution:
            # Use the explicit target; JSON serialization sorts dict keys, so we
            # must NOT rely on insertion order surviving a disk round-trip.
            target = md.get("target") or next(iter(solution))
            room = solution[target]["room"]
            obj = solution[target]["object"]
            ok = _contains(answer, room) and _contains(answer, obj)
            return RewardResult(1.0 if ok else 0.0, ok, f"need {room}, {obj}")
        return RewardResult(0.0, False, "no solution in metadata")

    return RewardResult(0.0, False, f"unscored memory task: {task}")


# --------------------------------------------------------------------------- #
# Formal logic (Fitch-style natural deduction)
# --------------------------------------------------------------------------- #

_PROOF_LINE = re.compile(r"^\s*(\d+)\.\s*(.+?)\s{2,}(\S.*?)\s*$")


def _norm_formula(text: str) -> str:
    return " ".join(text.split())


def _parse_proof(answer: str) -> list[tuple[int, str, str, list[int]]]:
    """Parse ``n. FORMULA    RULE i,j`` lines into (lineno, formula, rule, cited)."""
    lines: list[tuple[int, str, str, list[int]]] = []
    for raw in answer.splitlines():
        m = _PROOF_LINE.match(raw)
        if not m:
            continue
        lineno = int(m.group(1))
        formula = _norm_formula(m.group(2))
        rule = m.group(3).strip()
        cited = [int(d) for d in re.findall(r"\d+", rule)]
        lines.append((lineno, formula, rule, cited))
    return lines


def score_logic_proof(record: dict, output: str) -> RewardResult:
    """Structurally validate a Fitch proof against the goal/premises.

    Full soundness checking is out of scope; we reward three checkable
    properties that genuine proofs have and degenerate ones lack:
      * the goal formula is actually derived (0.5)
      * every stated premise appears as a Premise line (0.2)
      * every non-premise line cites only earlier line numbers (0.3)
    """
    md = record.get("metadata", {})
    goal = _norm_formula(md.get("goal", ""))
    premises = {_norm_formula(p) for p in md.get("premises", [])}
    lines = _parse_proof(final_answer(output))
    if not lines:
        return RewardResult(0.0, False, "no parseable proof lines")

    derived = {formula for _, formula, _, _ in lines}
    stated_premises = {f for _, f, rule, _ in lines if rule.lower().startswith("premise")}
    valid_linenos = {lineno for lineno, _, _, _ in lines}

    reached = bool(goal) and goal in derived
    premises_ok = premises.issubset(stated_premises)
    citations_ok = all(
        all(c in valid_linenos and c < lineno for c in cited)
        for lineno, _, rule, cited in lines
        if not rule.lower().startswith("premise")
    )

    score_val = (0.5 if reached else 0.0) + (0.2 if premises_ok else 0.0) + (0.3 if citations_ok else 0.0)
    correct = reached and premises_ok and citations_ok
    detail = f"goal={reached} premises={premises_ok} citations={citations_ok}"
    return RewardResult(score_val, correct, detail)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #

def score(record: dict, output: str) -> RewardResult:
    domain = record.get("domain")
    if domain == "chess":
        return score_chess_puzzle(record, output)
    if domain == "memory":
        return score_memory(record, output)
    if domain == "logic":
        if "goal" in record.get("metadata", {}):
            return score_logic_proof(record, output)
        # applied-reasoning / ethics are SFT-only: no programmatic verifier.
        return RewardResult(0.0, False, "no verifier for applied/ethics logic task")
    raise ValueError(f"no verifier registered for domain: {domain!r}")


def reward_value(record: dict, output: str) -> float:
    """Scalar reward for an RL loop (e.g. a TRL GRPO reward function)."""
    return score(record, output).score


# --------------------------------------------------------------------------- #
# TRL GRPO adapter (torch-free, so it stays unit-testable)
# --------------------------------------------------------------------------- #

def _completion_text(completion: object) -> str:
    """Extract text whether TRL hands us a string or a chat message list."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "\n".join(
            str(m.get("content", "")) for m in completion if isinstance(m, dict)
        )
    return str(completion)


def grpo_reward(completions: list, **columns: list) -> list[float]:
    """TRL GRPO reward function.

    TRL calls this with ``completions`` plus every dataset column as a parallel
    list. We carry ``domain``, ``task``, and ``meta`` (JSON) columns so the
    verifier can reconstruct a record and score each sampled generation.
    """
    n = len(completions)
    domains = columns.get("domain") or [None] * n
    tasks = columns.get("task") or [None] * n
    metas = columns.get("meta") or ["{}"] * n

    rewards: list[float] = []
    for i, comp in enumerate(completions):
        try:
            md = json.loads(metas[i]) if i < len(metas) and metas[i] else {}
        except (TypeError, ValueError):
            md = {}
        record = {
            "domain": domains[i] if i < len(domains) else None,
            "task": tasks[i] if i < len(tasks) else None,
            "metadata": md,
        }
        try:
            rewards.append(reward_value(record, _completion_text(comp)))
        except ValueError:
            rewards.append(0.0)
    return rewards

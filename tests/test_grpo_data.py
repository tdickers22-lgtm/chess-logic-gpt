from __future__ import annotations

import json

import chess

from chess_logic_gpt.chess.puzzles import make_puzzle_record
from chess_logic_gpt.logic.applied import generate_applied_reasoning_examples
from chess_logic_gpt.memory.generate import generate_memory_examples
from chess_logic_gpt.rewards import grpo_reward
from chess_logic_gpt.training.grpo_data import build_prompt_rows


def _render(messages: list[dict]) -> str:
    # Stand-in for tokenizer.apply_chat_template (no transformers needed here).
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages)


def _scholars_mate_puzzle() -> dict:
    # Lichess convention: moves[0] is the opponent's setup move, moves[1:] is the
    # solution. Black blunders ...Nf6, white mates with Qxf7#.
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"]:
        board.push_uci(uci)
    puzzle = make_puzzle_record("p1", board.fen(), ["g8f6", "h5f7"], 1200, ["mateIn1"])
    assert puzzle is not None
    return puzzle


def test_build_prompt_rows_filters_and_carries_columns() -> None:
    puzzle = _scholars_mate_puzzle()
    memory = next(
        r for r in generate_memory_examples(40, seed=3)
        if r["task"] == "working_memory_multi_query"
    )
    applied = generate_applied_reasoning_examples(1, seed=1)[0]  # not verifiable

    records = [puzzle, memory, applied]
    rows = build_prompt_rows(records, _render)

    # Non-verifiable applied-reasoning row is dropped; both verifiable rows kept.
    assert len(rows) == 2
    domains = {r["domain"] for r in rows}
    assert domains == {"chess", "memory"}
    for row in rows:
        assert set(row) == {"prompt", "domain", "task", "meta"}
        assert isinstance(row["meta"], str)  # JSON string, ready for the dataset column
        json.loads(row["meta"])  # parses


def test_grpo_reward_scores_gold_one_through_full_pipeline() -> None:
    # Exactly what TRL does: build prompt rows, then call grpo_reward with the
    # gold completions and the dataset columns. Gold must score 1.0.
    puzzle = _scholars_mate_puzzle()
    memory = next(
        r for r in generate_memory_examples(40, seed=3)
        if r["task"] == "constraint_reasoning_grid"
    )
    records = [puzzle, memory]
    rows = build_prompt_rows(records, _render)

    gold = [rec["messages"][-1]["content"] for rec in records]
    # Round-trip the columns through JSON (the dataset path) before scoring.
    rewards = grpo_reward(
        completions=gold,
        domain=[r["domain"] for r in rows],
        task=[r["task"] for r in rows],
        meta=[json.loads(json.dumps(r["meta"])) for r in rows],
    )
    assert rewards == [1.0, 1.0]

    # A wrong completion must score below 1.0 (real gradient signal exists).
    wrong = grpo_reward(
        completions=["<answer>totally wrong</answer>", "<answer>nonsense</answer>"],
        domain=[r["domain"] for r in rows],
        task=[r["task"] for r in rows],
        meta=[r["meta"] for r in rows],
    )
    assert all(x < 1.0 for x in wrong)

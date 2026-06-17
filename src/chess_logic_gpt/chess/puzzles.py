"""Ingest the Lichess open puzzle database (CC0) into training records.

The Lichess puzzle CSV (https://database.lichess.org/#puzzles) has columns:

    PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

Crucial convention: ``FEN`` is the position *before* the opponent's setup move,
and ``Moves`` is the full UCI line. The first move is the opponent's move that
creates the tactic; the solver is to move only *after* it. So we push the setup
move and store the resulting position as the puzzle FEN, with the remaining
moves as the forced solution line (alternating solver/reply). This is exactly
what ``rewards.score_chess_puzzle`` expects, and the themes feed
``curriculum.build_motif_curriculum``.

No engine is needed: the Lichess line is ground truth, and we re-validate every
move with python-chess so a malformed row is dropped rather than poisoning data.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path

import chess

from chess_logic_gpt.chess.curriculum import DEFAULT_MOTIF_PRIORITY, primary_motif
from chess_logic_gpt.records import stable_id
from chess_logic_gpt.training.trace import wrap_trace

LICHESS_PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"

_SYSTEM = (
    "You are a chess tactics engine. Find the best move for the side to move, "
    "verify the line is forced, and answer with the move(s) in standard algebraic notation."
)


def make_puzzle_record(
    puzzle_id: str,
    base_fen: str,
    moves_uci: list[str],
    rating: int,
    themes: list[str],
    priority: list[str] = DEFAULT_MOTIF_PRIORITY,
) -> dict | None:
    """Convert one Lichess puzzle row into a record, or None if malformed."""
    if len(moves_uci) < 2:
        return None
    try:
        board = chess.Board(base_fen)
    except ValueError:
        return None

    try:
        setup = chess.Move.from_uci(moves_uci[0])
    except ValueError:
        return None
    if setup not in board.legal_moves:
        return None
    board.push(setup)
    solver_fen = board.fen()

    line = moves_uci[1:]
    replay = chess.Board(solver_fen)
    san_line: list[str] = []
    for uci in line:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        if move not in replay.legal_moves:
            return None
        san_line.append(replay.san(move))
        replay.push(move)

    solver_sans = san_line[0::2]
    if not solver_sans:
        return None

    motif = primary_motif(themes, priority) or (themes[0] if themes else "tactic")
    side = "White" if chess.Board(solver_fen).turn == chess.WHITE else "Black"
    ascii_board = str(chess.Board(solver_fen))

    prompt = (
        "Solve the tactic. Give the best move for the side to move, then the rest of the forced line.\n\n"
        f"FEN: {solver_fen}\n\n{ascii_board}\n\n{side} to move."
    )
    reasoning = (
        f"Motif: {motif}. {side} to move. Calculate the forcing line: {' '.join(san_line)}. "
        f"The decisive idea is the {motif}; quiet alternatives let the opponent escape."
    )
    answer = " ".join(solver_sans)

    return {
        "id": stable_id("lichess-puzzle", puzzle_id),
        "domain": "chess",
        "task": "chess_tactic_solve",
        "source": {
            "name": "lichess_puzzles",
            "url": "https://database.lichess.org/#puzzles",
            "license": "CC0-1.0",
            "provenance": "lichess-open-database",
        },
        "messages": [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": wrap_trace(reasoning, answer)},
        ],
        "verification": {"status": "verified", "method": "lichess-solution-line"},
        "metadata": {
            "puzzle_id": puzzle_id,
            "fen": solver_fen,
            "line_uci": line,
            "themes": themes,
            "primary_motif": motif,
            "rating": rating,
        },
    }


def _open_text(path: str | Path) -> io.TextIOBase:
    """Open a .csv or .csv.zst file as a UTF-8 text stream."""
    path = Path(path)
    if path.suffix == ".zst":
        import zstandard

        fh = path.open("rb")
        reader = zstandard.ZstdDecompressor().stream_reader(fh)
        return io.TextIOWrapper(reader, encoding="utf-8")
    return path.open("r", encoding="utf-8")


def iter_lichess_rows(
    path: str | Path,
    *,
    limit: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    require_motifs: set[str] | None = None,
) -> Iterator[dict]:
    """Stream parsed puzzle rows from a Lichess CSV (plain or .zst)."""
    stream = _open_text(path)
    try:
        reader = csv.DictReader(stream)
        emitted = 0
        for row in reader:
            try:
                rating = int(row.get("Rating", "0") or 0)
            except ValueError:
                continue
            if min_rating is not None and rating < min_rating:
                continue
            if max_rating is not None and rating > max_rating:
                continue
            themes = (row.get("Themes") or "").split()
            if require_motifs and not (set(themes) & require_motifs):
                continue
            moves = (row.get("Moves") or "").split()
            yield {
                "puzzle_id": row.get("PuzzleId", ""),
                "fen": row.get("FEN", ""),
                "moves": moves,
                "rating": rating,
                "themes": themes,
            }
            emitted += 1
            if limit is not None and emitted >= limit:
                return
    finally:
        stream.close()


def puzzles_from_csv(
    path: str | Path,
    *,
    limit: int | None = None,
    min_rating: int | None = None,
    max_rating: int | None = None,
    require_motifs: set[str] | None = None,
) -> list[dict]:
    """Parse + validate a Lichess CSV into clean chess training records."""
    records: list[dict] = []
    for row in iter_lichess_rows(
        path,
        limit=limit,
        min_rating=min_rating,
        max_rating=max_rating,
        require_motifs=require_motifs,
    ):
        record = make_puzzle_record(
            row["puzzle_id"], row["fen"], row["moves"], row["rating"], row["themes"]
        )
        if record is not None:
            records.append(record)
    return records


def harvest_motif_pool(
    path: str | Path,
    *,
    motifs: list[str],
    cap_distinct: int,
    min_rating: int | None = None,
    max_rating: int | None = None,
    priority: list[str] = DEFAULT_MOTIF_PRIORITY,
) -> list[dict]:
    """Stream the (huge) CSV and collect up to ``cap_distinct`` puzzles per motif.

    Memory-bounded: only ``len(motifs) * cap_distinct`` records are ever held, and
    a row is validated with python-chess only when its motif's bucket isn't full
    yet -- so once common motifs fill, the scan races ahead cheaply to gather the
    rare ones (e.g. backRankMate). Stops early once every motif's bucket is full.
    """
    motifset = set(motifs)
    buckets: dict[str, list[dict]] = {m: [] for m in motifs}
    seen_fen: dict[str, set[str]] = {m: set() for m in motifs}
    full: set[str] = set()

    for row in iter_lichess_rows(path, min_rating=min_rating, max_rating=max_rating):
        if len(full) == len(motifs):
            break
        motif = primary_motif(row["themes"], priority)
        if motif not in motifset or motif in full:
            continue
        record = make_puzzle_record(
            row["puzzle_id"], row["fen"], row["moves"], row["rating"], row["themes"], priority
        )
        if record is None:
            continue
        fen = record["metadata"]["fen"]
        if fen in seen_fen[motif]:
            continue
        seen_fen[motif].add(fen)
        buckets[motif].append(record)
        if len(buckets[motif]) >= cap_distinct:
            full.add(motif)

    return [record for motif in motifs for record in buckets[motif]]

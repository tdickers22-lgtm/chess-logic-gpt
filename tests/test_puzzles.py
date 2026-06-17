from __future__ import annotations

import csv
from pathlib import Path

import chess

from chess_logic_gpt.chess.puzzles import (
    harvest_motif_pool,
    make_puzzle_record,
    puzzles_from_csv,
)
from chess_logic_gpt.rewards import final_answer, score


def _scholars_mate_setup() -> tuple[str, list[str]]:
    """Position before 3...Nf6?? with the solution 4.Qxf7#.

    Lichess convention: FEN is *before* the opponent's setup move, Moves is the
    full UCI line (setup first, then the solver's reply).
    """
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"]:
        board.push_uci(uci)
    return board.fen(), ["g8f6", "h5f7"]


def test_make_puzzle_record_picks_motif_and_solver_position() -> None:
    base_fen, moves = _scholars_mate_setup()
    record = make_puzzle_record("testpuz", base_fen, moves, 1500, ["mateIn1", "fork"])
    assert record is not None
    assert record["domain"] == "chess"
    assert record["task"] == "chess_tactic_solve"
    assert record["metadata"]["primary_motif"] == "mateIn1"
    assert record["metadata"]["line_uci"] == ["h5f7"]

    # The stored solver FEN + line really is mate.
    board = chess.Board(record["metadata"]["fen"])
    for uci in record["metadata"]["line_uci"]:
        board.push_uci(uci)
    assert board.is_checkmate()


def test_puzzle_record_is_self_consistent_under_verifier() -> None:
    base_fen, moves = _scholars_mate_setup()
    record = make_puzzle_record("testpuz", base_fen, moves, 1500, ["mateIn1"])
    assert record is not None
    gold = record["messages"][2]["content"]
    result = score(record, gold)
    assert result.correct
    assert result.score == 1.0
    # And the gold answer is the SAN mate move.
    assert final_answer(gold).strip() == "Qxf7#"


def test_wrong_move_scores_zero() -> None:
    base_fen, moves = _scholars_mate_setup()
    record = make_puzzle_record("testpuz", base_fen, moves, 1500, ["mateIn1"])
    assert record is not None
    assert score(record, "<answer>Qh4</answer>").score == 0.0


def test_malformed_rows_are_dropped() -> None:
    base_fen, _ = _scholars_mate_setup()
    # Illegal setup move -> None
    assert make_puzzle_record("bad", base_fen, ["a1a8", "h5f7"], 1500, ["fork"]) is None
    # Too few moves -> None
    assert make_puzzle_record("bad", base_fen, ["g8f6"], 1500, ["fork"]) is None


def test_puzzles_from_csv_roundtrip(tmp_path: Path) -> None:
    base_fen, moves = _scholars_mate_setup()
    csv_path = tmp_path / "puzzles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
             "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"]
        )
        writer.writerow(
            ["testpuz", base_fen, " ".join(moves), "1500", "80",
             "95", "1200", "mateIn1 fork", "https://lichess.org/x", ""]
        )

    records = puzzles_from_csv(csv_path)
    assert len(records) == 1
    assert records[0]["metadata"]["primary_motif"] == "mateIn1"
    assert score(records[0], records[0]["messages"][2]["content"]).score == 1.0


def _fen_after(ucis: list[str]) -> str:
    board = chess.Board()
    for uci in ucis:
        board.push_uci(uci)
    return board.fen()


def _write_csv(tmp_path: Path, rows: list[list[str]]) -> Path:
    path = tmp_path / "p.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["PuzzleId", "FEN", "Moves", "Rating", "RatingDeviation",
             "Popularity", "NbPlays", "Themes", "GameUrl", "OpeningTags"]
        )
        for row in rows:
            writer.writerow(row)
    return path


def test_harvest_motif_pool_dedupes_and_caps(tmp_path: Path) -> None:
    f_scholar, _ = _scholars_mate_setup()
    f_e4 = _fen_after(["e2e4"])
    f_d4 = _fen_after(["d2d4"])
    rows = [
        ["p1", f_scholar, "g8f6 h5f7", "1500", "80", "90", "100", "fork", "u", ""],
        ["p1dup", f_scholar, "g8f6 h5f7", "1600", "80", "90", "100", "fork", "u", ""],  # same FEN
        ["p2", f_e4, "e7e5 g1f3", "1400", "80", "90", "100", "fork", "u", ""],
        ["p3", f_d4, "d7d5 c1f4", "1400", "80", "90", "100", "pin", "u", ""],
    ]
    csv_path = _write_csv(tmp_path, rows)

    pool = harvest_motif_pool(csv_path, motifs=["fork", "pin"], cap_distinct=5)
    forks = [r for r in pool if r["metadata"]["primary_motif"] == "fork"]
    pins = [r for r in pool if r["metadata"]["primary_motif"] == "pin"]
    assert len(forks) == 2  # duplicate FEN dropped
    assert len(pins) == 1

    capped = harvest_motif_pool(csv_path, motifs=["fork", "pin"], cap_distinct=1)
    assert sum(1 for r in capped if r["metadata"]["primary_motif"] == "fork") == 1

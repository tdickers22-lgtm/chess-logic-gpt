from __future__ import annotations

import chess
import pytest

from chess_logic_gpt.rewards import (
    extract_tag,
    final_answer,
    parse_move_line,
    reward_value,
    score,
)


def _fen_after(ucis: list[str]) -> str:
    board = chess.Board()
    for u in ucis:
        board.push_uci(u)
    return board.fen()


# --- trace parsing --------------------------------------------------------- #

def test_extract_tag_and_final_answer():
    text = "<reasoning>fork on f7</reasoning>\n<answer>Qxf7#</answer>"
    assert extract_tag(text, "reasoning") == "fork on f7"
    assert final_answer(text) == "Qxf7#"


def test_final_answer_falls_back_to_whole_text():
    assert final_answer("  Qxf7#  ") == "Qxf7#"


# --- chess tactics --------------------------------------------------------- #

# Scholar's mate: after 1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6, White mates with Qxf7#.
SCHOLAR_SETUP = ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6"]
MATE_FEN = _fen_after(SCHOLAR_SETUP)


def test_mate_in_one_fen_is_actually_mate():
    board = chess.Board(MATE_FEN)
    board.push_uci("h5f7")
    assert board.is_checkmate()


def _puzzle(line_uci: list[str], fen: str = MATE_FEN) -> dict:
    return {"domain": "chess", "task": "chess_tactic_solve",
            "metadata": {"fen": fen, "line_uci": line_uci}}


def test_chess_correct_move_san_and_uci():
    rec = _puzzle(["h5f7"])
    assert score(rec, "<answer>Qxf7#</answer>").correct
    assert score(rec, "<answer>h5f7</answer>").correct
    assert reward_value(rec, "<answer>Qxf7#</answer>") == 1.0


def test_chess_wrong_legal_move_scores_zero():
    rec = _puzzle(["h5f7"])
    result = score(rec, "<answer>Ng1f3</answer>")  # legal but not the solution
    assert result.score == 0.0
    assert not result.correct


def test_chess_illegal_move_scores_zero():
    rec = _puzzle(["h5f7"])
    # e2e4 is illegal here (the e-pawn is already on e4); it must not be credited.
    assert score(rec, "<answer>e2e4</answer>").score == 0.0


def test_chess_partial_credit_on_multi_move_line():
    # A 2-solver-move puzzle: solver plays s1, opponent forced reply r1, solver s2.
    s1, r1, s2 = "h5f7", "e8f7", "c4d5"  # illustrative; only s1 correctness is asserted
    rec = _puzzle([s1, r1, s2])
    only_first = score(rec, "<answer>h5f7</answer>")
    assert only_first.score == 0.5
    assert not only_first.correct


def test_chess_accepts_full_line_with_move_numbers():
    rec = _puzzle(["h5f7"])
    assert score(rec, "<answer>1. Qxf7#</answer>").correct


def test_chess_two_solver_move_line_scores_full():
    # Real mateIn2 where every ply targets d8 (Rb8+ / ...Rxd8 / Rxd8#): the
    # solver-only gold answer must self-score 1.0, and the full line is accepted.
    rec = _puzzle(["b7b8", "d7d8", "b8d8"], fen="6k1/1Rpr1ppp/8/2p1P3/r2n1PP1/p7/K1P4P/3R4 w - - 2 27")
    assert score(rec, "<answer>Rb8+ Rxd8#</answer>").score == 1.0
    assert score(rec, "<answer>Rb8+ Rxd8 Rxd8#</answer>").correct
    # First move only -> half credit (one of two solver moves).
    assert score(rec, "<answer>Rb8+</answer>").score == 0.5


def test_parse_move_line_skips_prose():
    moves = parse_move_line(MATE_FEN, "I see a mate: Qxf7# wins on the spot")
    assert [m.uci() for m in moves] == ["h5f7"]


# --- memory ---------------------------------------------------------------- #

def test_memory_fact_recall():
    rec = {
        "domain": "memory",
        "task": "working_memory_fact_recall",
        "metadata": {
            "facts": ["Gideon has the black notebook.", "Hana has the yellow bishop."],
            "query": "Gideon",
        },
    }
    assert score(rec, "<answer>Gideon has the black notebook.</answer>").correct
    assert not score(rec, "<answer>Gideon has the yellow bishop.</answer>").correct


def test_memory_constraint_grid():
    rec = {
        "domain": "memory",
        "task": "constraint_reasoning_grid",
        "metadata": {"solution": {"Ada": {"room": "library", "object": "rook"},
                                  "Benoit": {"room": "kitchen", "object": "coin"}}},
    }
    assert score(rec, "<answer>Ada is in the library and has the rook.</answer>").correct
    assert not score(rec, "<answer>Ada is in the kitchen and has the coin.</answer>").correct


def test_memory_multi_query_is_graded():
    rec = {
        "domain": "memory",
        "task": "working_memory_multi_query",
        "metadata": {"needed": ["red rook", "blue coin", "green key"]},
    }
    full = score(rec, "<answer>Ada has the red rook. Bo has the blue coin. Cy has the green key.</answer>")
    assert full.score == 1.0 and full.correct
    partial = score(rec, "<answer>Ada has the red rook. Bo has the blue coin.</answer>")
    assert partial.score == pytest.approx(2 / 3) and not partial.correct
    miss = score(rec, "<answer>nothing relevant here.</answer>")
    assert miss.score == 0.0

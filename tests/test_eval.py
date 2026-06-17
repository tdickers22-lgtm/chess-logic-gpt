from __future__ import annotations

import chess

from chess_logic_gpt.chess.puzzles import make_puzzle_record
from chess_logic_gpt.eval import evaluate, gold_answer, is_verifiable
from chess_logic_gpt.logic.generate import generate_logic_examples
from chess_logic_gpt.memory.generate import generate_memory_examples


def _puzzle() -> dict:
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"]:
        board.push_uci(uci)
    record = make_puzzle_record("p", board.fen(), ["g8f6", "h5f7"], 1500, ["mateIn1"])
    assert record is not None
    return record


def test_is_verifiable_gates_domains() -> None:
    assert is_verifiable({"domain": "chess", "metadata": {}})
    assert is_verifiable({"domain": "memory", "metadata": {}})
    assert is_verifiable({"domain": "logic", "metadata": {"goal": "Q"}})
    assert not is_verifiable({"domain": "logic", "metadata": {}})


def test_perfect_generate_scores_one_across_domains() -> None:
    records = (
        generate_memory_examples(6, seed=1)
        + generate_logic_examples(4, seed=1)
        + [_puzzle()]
    )
    report = evaluate(records, gold_answer)
    assert report["overall"]["accuracy"] == 1.0
    for stats in report["by_domain"].values():
        assert stats["accuracy"] == 1.0
    assert "mateIn1" in report["by_motif"]
    assert report["memory_recall_by_facts"]  # at least one fact-count bucket


def test_empty_generate_scores_zero() -> None:
    records = generate_memory_examples(6, seed=2) + [_puzzle()]
    report = evaluate(records, lambda r: "")
    assert report["overall"]["accuracy"] == 0.0
    assert report["overall"]["n"] == 7

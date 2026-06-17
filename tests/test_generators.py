from __future__ import annotations

import json

import chess

from chess_logic_gpt.chess.endgames import generate_endgame_candidates
from chess_logic_gpt.logic.applied import generate_applied_reasoning_examples
from chess_logic_gpt.logic.ethics_calibration import generate_ethics_calibration_examples
from chess_logic_gpt.logic.generate import generate_logic_examples
from chess_logic_gpt.memory.generate import generate_memory_examples
from chess_logic_gpt.rewards import score
from chess_logic_gpt.training.formatting import row_to_text


def assert_record_shape(row: dict, domain: str) -> None:
    assert row["id"]
    assert row["domain"] == domain
    assert row["task"]
    assert row["source"]["name"]
    assert row["source"]["license"]
    assert row["verification"]["status"]
    assert len(row["messages"]) == 3
    assert row["messages"][0]["role"] == "system"
    assert row["messages"][1]["role"] == "user"
    assert row["messages"][2]["role"] == "assistant"


def test_logic_generator_record_shape() -> None:
    rows = generate_logic_examples(20, seed=10)
    assert len(rows) == 20
    for row in rows:
        assert_record_shape(row, "logic")
        assert "Proof:" in row["messages"][2]["content"]


def test_applied_reasoning_generator_record_shape() -> None:
    rows = generate_applied_reasoning_examples(20, seed=11)
    assert len(rows) == 20
    tasks = {row["task"] for row in rows}
    assert all(task.startswith("applied_") for task in tasks)
    assert len(tasks) >= 3
    for row in rows:
        assert_record_shape(row, "logic")
        answer = row["messages"][2]["content"].lower()
        assert any(term in answer for term in ["assumption", "evidence", "counterexample", "failure", "truth", "valid"])


def test_ethics_calibration_generator_record_shape() -> None:
    rows = generate_ethics_calibration_examples(60, seed=15)
    assert len(rows) == 60
    decisions = {row["metadata"]["decision"] for row in rows}
    assert "answer" in decisions
    assert "refuse_specific_harmful_request" in decisions
    for row in rows:
        assert_record_shape(row, "logic")
        assert row["source"]["name"] == "generated_ethics_calibration"
        answer = row["messages"][2]["content"].lower()
        assert any(term in answer for term in ["allowed", "cannot", "ethic", "refusal", "answer"])


def test_memory_generator_record_shape() -> None:
    rows = generate_memory_examples(20, seed=12)
    assert len(rows) == 20
    for row in rows:
        assert_record_shape(row, "memory")


def test_memory_gold_answers_self_score() -> None:
    # Every generated memory example's own gold answer must score a perfect 1.0,
    # otherwise the RL/eval reward signal is corrupted. Score AFTER a JSON
    # round-trip (sort_keys mirrors write_jsonl) so scoring cannot secretly
    # depend on dict insertion order, which disk serialization does not preserve.
    rows = generate_memory_examples(60, seed=7)
    tasks_seen = set()
    for row in rows:
        roundtripped = json.loads(json.dumps(row, sort_keys=True))
        gold = roundtripped["messages"][2]["content"]
        result = score(roundtripped, gold)
        assert result.score == 1.0, (row["task"], result.detail, row["metadata"])
        tasks_seen.add(row["task"])
    assert {
        "working_memory_fact_recall",
        "working_memory_multi_query",
        "working_memory_order_transform",
        "constraint_reasoning_grid",
    } <= tasks_seen


def test_endgame_candidates_are_legal_low_piece_positions() -> None:
    rows = generate_endgame_candidates(20, seed=13)
    assert len(rows) == 20
    for row in rows:
        assert_record_shape(row, "chess")
        board = chess.Board(row["metadata"]["fen"])
        assert board.status() == chess.STATUS_VALID
        assert len(board.piece_map()) <= 7
        assert row["verification"]["status"] == "unverified"


def test_row_to_text_uses_messages() -> None:
    row = generate_applied_reasoning_examples(1, seed=14)[0]
    text = row_to_text(row)
    assert "<|system|>" in text
    assert "<|user|>" in text
    assert "<|assistant|>" in text

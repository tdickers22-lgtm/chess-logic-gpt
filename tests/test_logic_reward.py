from __future__ import annotations

import json

import chess

from chess_logic_gpt.chess.puzzles import make_puzzle_record
from chess_logic_gpt.logic.generate import generate_logic_examples
from chess_logic_gpt.rewards import grpo_reward, score, score_logic_proof


def test_generated_logic_proof_scores_full() -> None:
    for record in generate_logic_examples(20, seed=3):
        gold = record["messages"][2]["content"]
        result = score(record, gold)
        assert result.correct, result.detail
        assert result.score == 1.0


def test_missing_goal_line_loses_credit() -> None:
    record = {
        "domain": "logic",
        "task": "fitch_modus_ponens",
        "metadata": {"premises": ["P -> Q", "P"], "goal": "Q"},
    }
    # Cites a premise but never derives the goal Q.
    broken = "<answer>Proof:\n1. P -> Q    Premise\n2. P    Premise\n</answer>"
    result = score_logic_proof(record, broken)
    assert not result.correct
    assert 0.0 < result.score < 1.0


def test_hallucinated_citation_is_penalised() -> None:
    record = {
        "domain": "logic",
        "task": "fitch_modus_ponens",
        "metadata": {"premises": ["P -> Q", "P"], "goal": "Q"},
    }
    # Goal reached and premises stated, but line 3 cites a nonexistent line 9.
    proof = "<answer>Proof:\n1. P -> Q    Premise\n2. P    Premise\n3. Q    ->E 1,9</answer>"
    result = score_logic_proof(record, proof)
    assert not result.correct
    assert result.score < 1.0


def test_applied_logic_has_no_verifier() -> None:
    record = {"domain": "logic", "task": "applied_argument_audit", "metadata": {}}
    result = score(record, "<answer>anything</answer>")
    assert not result.correct
    assert result.score == 0.0


def test_grpo_reward_batches_chess_strings_and_chat() -> None:
    board = chess.Board()
    for uci in ["e2e4", "e7e5", "f1c4", "b8c6", "d1h5"]:
        board.push_uci(uci)
    record = make_puzzle_record("p", board.fen(), ["g8f6", "h5f7"], 1500, ["mateIn1"])
    assert record is not None
    meta = json.dumps(
        {"fen": record["metadata"]["fen"], "line_uci": record["metadata"]["line_uci"]}
    )
    completions = [
        "<answer>Qxf7#</answer>",                                    # plain string
        [{"role": "assistant", "content": "<answer>Qh4</answer>"}],  # chat list
    ]
    rewards = grpo_reward(
        completions,
        domain=["chess", "chess"],
        task=["chess_tactic_solve", "chess_tactic_solve"],
        meta=[meta, meta],
    )
    assert rewards == [1.0, 0.0]

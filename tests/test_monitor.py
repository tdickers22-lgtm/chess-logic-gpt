from __future__ import annotations

from pathlib import Path

from chess_logic_gpt.training.monitoring import (
    MetricLogger,
    evaluate_guardrails,
    read_metrics_jsonl,
    worst_level,
)


def _codes(issues) -> set[str]:
    return {i.code for i in issues}


def test_healthy_history_has_no_issues() -> None:
    history = [
        {"step": i, "loss": 1.0 - 0.001 * i, "reward": 0.4 + 0.001 * i, "kl": 0.05, "grad_norm": 1.0}
        for i in range(60)
    ]
    assert evaluate_guardrails(history) == []


def test_nan_loss_is_error() -> None:
    issues = evaluate_guardrails([{"loss": float("nan")}])
    assert "nan_loss" in _codes(issues)
    assert worst_level(issues) == "ERROR"


def test_reward_collapse_and_floor() -> None:
    history = [{"reward": 0.6} for _ in range(20)] + [{"reward": 0.0} for _ in range(20)]
    codes = _codes(evaluate_guardrails(history))
    assert "reward_collapse" in codes
    assert "reward_floor" in codes


def test_grad_explosion_and_kl_blowup() -> None:
    codes = _codes(evaluate_guardrails([{"grad_norm": 250.0, "kl": 1.2}]))
    assert "grad_explosion" in codes
    assert "kl_blowup" in codes


def test_loss_spike_and_stall() -> None:
    spike = [{"loss": 1.0} for _ in range(25)] + [{"loss": 10.0}]
    assert "loss_spike" in _codes(evaluate_guardrails(spike))
    stall = [{"loss": 0.5} for _ in range(25)]
    assert "loss_stall" in _codes(evaluate_guardrails(stall))


def test_eval_regression_and_reward_hacking() -> None:
    assert "eval_regression" in _codes(
        evaluate_guardrails([{"eval_accuracy": 0.7}, {"eval_accuracy": 0.6}])
    )
    hacking = [{"reward": 0.99, "eval_accuracy": 0.2} for _ in range(20)]
    assert "reward_eval_divergence" in _codes(evaluate_guardrails(hacking))


def test_metric_logger_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    logger = MetricLogger(path, use_trackio=False)
    logger.log({"step": 1, "loss": 0.5})
    logger.log({"step": 2, "loss": 0.4})
    logger.finish()
    history = read_metrics_jsonl(path)
    assert [row["step"] for row in history] == [1, 2]

"""Training telemetry + guardrails for autonomous monitoring.

Two layers, both torch-free so they unit-test and run anywhere:

* ``MetricLogger`` — the trainer logs each step's metrics here. It appends a
  local ``metrics.jsonl`` (always works, even on a detached cloud box) and, if
  Trackio is installed/configured, mirrors to a Trackio dashboard + alerts.
* ``evaluate_guardrails`` — a pure function over the metric history that returns
  ``Issue``s (NaN loss, reward collapse, KL blowup, grad explosion, loss spike,
  stall, eval regression, reward/eval divergence a.k.a. reward hacking).

``scripts/monitor_training.py`` is the actual monitoring agent: it tails the
metric history, runs the guardrails on a loop, fires alerts, and can escalate to
a Cursor background agent for diagnosis.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

LEVELS = ("INFO", "WARN", "ERROR")

DEFAULT_THRESHOLDS: dict[str, float] = {
    "reward_floor": 0.05,        # mean reward below this in the recent window is bad
    "reward_collapse_drop": 0.15,  # recent window mean dropped this much vs the prior window
    "kl_max": 0.5,               # GRPO KL to reference above this is runaway
    "grad_norm_max": 100.0,      # exploding gradients
    "loss_spike_factor": 3.0,    # loss > factor * recent median
    "loss_stall_eps": 1e-4,      # max-min loss over the window below this == stalled
    "eval_regression_drop": 0.05,  # eval accuracy fell this far below the best seen
    "reward_hack_reward": 0.98,  # near-perfect reward...
    "reward_hack_eval": 0.5,     # ...while eval accuracy is this low == suspicious
}


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def _series(history: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out.append(float(value))
    return out


def evaluate_guardrails(
    history: list[dict],
    *,
    thresholds: dict[str, float] | None = None,
    window: int = 20,
) -> list[Issue]:
    """Inspect the metric history and return any triggered guardrails."""
    t = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    issues: list[Issue] = []
    if not history:
        return issues
    last = history[-1]

    loss = last.get("loss")
    if isinstance(loss, (int, float)):
        if math.isnan(loss) or math.isinf(loss):
            issues.append(Issue("ERROR", "nan_loss", f"loss is {loss} at the latest step"))

    losses = _series(history, "loss")
    if len(losses) >= window + 1:
        recent = losses[-(window + 1):-1]
        med = median(recent)
        if med > 0 and losses[-1] > t["loss_spike_factor"] * med:
            issues.append(
                Issue("WARN", "loss_spike", f"loss {losses[-1]:.4f} > {t['loss_spike_factor']}x median {med:.4f}")
            )
    if len(losses) >= window:
        window_losses = losses[-window:]
        if max(window_losses) - min(window_losses) < t["loss_stall_eps"]:
            issues.append(Issue("INFO", "loss_stall", f"loss flat (<{t['loss_stall_eps']}) over {window} logs"))

    rewards = _series(history, "reward")
    if len(rewards) >= 2 * window:
        recent_mean = mean(rewards[-window:])
        prior_mean = mean(rewards[-2 * window:-window])
        if recent_mean < prior_mean - t["reward_collapse_drop"]:
            issues.append(
                Issue("WARN", "reward_collapse", f"reward fell {prior_mean:.3f}->{recent_mean:.3f}")
            )
    if rewards and mean(rewards[-window:]) < t["reward_floor"] and len(rewards) >= window:
        issues.append(Issue("WARN", "reward_floor", f"mean reward {mean(rewards[-window:]):.3f} near zero"))

    kl = last.get("kl")
    if isinstance(kl, (int, float)) and kl > t["kl_max"]:
        issues.append(Issue("WARN", "kl_blowup", f"KL {kl:.3f} > {t['kl_max']}"))

    grad = last.get("grad_norm")
    if isinstance(grad, (int, float)) and grad > t["grad_norm_max"]:
        issues.append(Issue("ERROR", "grad_explosion", f"grad_norm {grad:.1f} > {t['grad_norm_max']}"))

    evals = _series(history, "eval_accuracy")
    if len(evals) >= 2:
        best = max(evals[:-1])
        if evals[-1] < best - t["eval_regression_drop"]:
            issues.append(
                Issue("WARN", "eval_regression", f"eval acc {evals[-1]:.3f} below best {best:.3f}")
            )

    if rewards and evals:
        if mean(rewards[-window:]) > t["reward_hack_reward"] and evals[-1] < t["reward_hack_eval"]:
            issues.append(
                Issue(
                    "WARN",
                    "reward_eval_divergence",
                    f"reward ~{mean(rewards[-window:]):.2f} but eval acc {evals[-1]:.2f} (possible reward hacking)",
                )
            )

    return issues


def worst_level(issues: list[Issue]) -> str | None:
    for level in ("ERROR", "WARN", "INFO"):
        if any(i.level == level for i in issues):
            return level
    return None


# --------------------------------------------------------------------------- #
# Trackio (soft dependency)
# --------------------------------------------------------------------------- #

def _try_trackio():
    try:
        import trackio
    except Exception:
        return None
    return trackio


class MetricLogger:
    """Append metrics to a local JSONL and mirror to Trackio when available."""

    def __init__(
        self,
        path: str | Path,
        *,
        project: str = "chess-logic-gpt",
        run: str | None = None,
        space_id: str | None = None,
        config: dict[str, Any] | None = None,
        use_trackio: bool = True,
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.trackio = _try_trackio() if use_trackio else None
        if self.trackio is not None:
            try:
                self.trackio.init(project=project, name=run, space_id=space_id, config=config or {})
            except Exception:
                self.trackio = None

    def log(self, row: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        if self.trackio is not None:
            try:
                self.trackio.log(row)
            except Exception:
                pass

    def alert(self, title: str, text: str, level: str = "WARN") -> None:
        line = f"[{level}] {title}: {text}"
        print(line, flush=True)
        if self.trackio is not None:
            try:
                self.trackio.alert(title=title, text=text, level=getattr(self.trackio.AlertLevel, level, None))
            except Exception:
                pass

    def finish(self) -> None:
        if self.trackio is not None:
            try:
                self.trackio.finish()
            except Exception:
                pass


def read_metrics_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

"""Transformers/TRL callback that wires guardrails into the training loop.

Imported only by the GPU training scripts (it needs `transformers`), so the
torch-free monitoring core stays unit-testable on its own. On every log it:
  * records the step's metrics to the ``MetricLogger`` (local JSONL + Trackio),
  * runs ``evaluate_guardrails`` over the running history,
  * fires a Trackio alert the first time each guardrail trips,
  * stops training on any ERROR-level guardrail, and
  * honours an external STOP sentinel file so the monitor agent has a kill switch.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from transformers import TrainerCallback

from chess_logic_gpt.training.monitoring import MetricLogger, evaluate_guardrails


class GuardrailCallback(TrainerCallback):
    def __init__(self, logger: MetricLogger, *, window: int = 20, stop_file: str | None = None) -> None:
        self.logger = logger
        self.window = window
        self.history: list[dict] = []
        self._alerted: set[str] = set()
        self.stop_file = Path(stop_file) if stop_file else None

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: ANN001
        if not logs:
            return control
        row = {k: v for k, v in logs.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
        row["step"] = int(state.global_step)
        self.logger.log(row)
        self.history.append(row)
        for issue in evaluate_guardrails(self.history, window=self.window):
            if issue.level in ("WARN", "ERROR") and issue.code not in self._alerted:
                self.logger.alert(issue.code, issue.message, level=issue.level)
                self._alerted.add(issue.code)
            if issue.level == "ERROR":
                control.should_training_stop = True
        return control

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001
        if self.stop_file is not None and self.stop_file.exists():
            self.logger.alert("external_stop", "STOP sentinel present; halting", level="ERROR")
            control.should_training_stop = True
        return control

    def on_train_end(self, args, state, control, **kwargs):  # noqa: ANN001
        self.logger.finish()
        return control


class CheckpointMirrorCallback(TrainerCallback):
    """Copy each completed Trainer checkpoint to a second location.

    Colab local disk can disappear when a runtime dies. Mirroring every checkpoint
    to Drive as soon as it is saved is more robust than waiting for a final
    notebook cell after training exits.
    """

    def __init__(self, mirror_dir: str | Path, *, keep: int = 5) -> None:
        self.mirror_dir = Path(mirror_dir)
        self.keep = keep

    @staticmethod
    def _checkpoint_step(path: Path) -> int:
        try:
            return int(path.name.split("-")[-1])
        except ValueError:
            return -1

    def _prune(self) -> None:
        checkpoints = sorted(
            (p for p in self.mirror_dir.glob("checkpoint-*") if p.is_dir()),
            key=self._checkpoint_step,
        )
        for old in checkpoints[: max(0, len(checkpoints) - self.keep)]:
            shutil.rmtree(old, ignore_errors=True)

    def on_save(self, args, state, control, **kwargs):  # noqa: ANN001
        if state.global_step <= 0:
            return control
        src = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        if not src.is_dir():
            return control
        self.mirror_dir.mkdir(parents=True, exist_ok=True)
        dest = self.mirror_dir / src.name
        tmp = self.mirror_dir / f".{src.name}.tmp"
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
            shutil.copytree(src, tmp)
            if dest.exists():
                shutil.rmtree(dest)
            tmp.rename(dest)
            self._prune()
            print(f"Mirrored checkpoint -> {dest}", flush=True)
        except Exception as exc:
            print(f"checkpoint mirror failed for {src}: {exc}", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
        return control

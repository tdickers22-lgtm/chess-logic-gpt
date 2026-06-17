#!/usr/bin/env python3
"""Autonomous training-monitor agent.

Tails the trainer's ``metrics.jsonl`` (written by every run, locally or on a
Modal volume), runs the guardrails on a loop, prints/forwards alerts, and on any
ERROR writes a ``STOP`` sentinel that the trainer's ``GuardrailCallback`` honours
as a kill switch. With ``--escalate`` (and ``CURSOR_API_KEY`` + ``cursor-agent``
installed) it spins up a Cursor background agent to diagnose the failure.

Examples
--------
    # one-shot check (CI / cron)
    python scripts/monitor_training.py --metrics-file data/outputs/qwen-chess-logic-grpo/metrics.jsonl --once

    # continuous watch with auto-stop + agent escalation
    python scripts/monitor_training.py \
        --metrics-file data/outputs/qwen-chess-logic-grpo/metrics.jsonl \
        --interval 30 --escalate
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import time
from pathlib import Path

from chess_logic_gpt.training.monitoring import (
    Issue,
    evaluate_guardrails,
    read_metrics_jsonl,
    worst_level,
)


def escalate(issues: list[Issue], metrics_file: str) -> None:
    if not (os.environ.get("CURSOR_API_KEY") and shutil.which("cursor-agent")):
        print("[monitor] escalation requested but disabled "
              "(set CURSOR_API_KEY and install cursor-agent to enable)")
        return
    summary = "; ".join(f"{i.level} {i.code}: {i.message}" for i in issues)
    prompt = (
        f"Training guardrails tripped for chess-logic-gpt: {summary}. "
        f"Inspect the metric history at {metrics_file} and the training logs, "
        f"diagnose the most likely root cause, and recommend a concrete fix "
        f"(hyperparameter change, data problem, or whether to stop the run)."
    )
    try:
        subprocess.Popen(["cursor-agent", "-p", prompt, "--output-format", "text"])  # noqa: S603,S607
        print("[monitor] escalated to a Cursor background agent")
    except OSError as exc:
        print(f"[monitor] escalation failed: {exc}")


def run_once(metrics_file: str, stop_file: str, window: int, do_escalate: bool) -> str | None:
    history = read_metrics_jsonl(metrics_file)
    if not history:
        print(f"[monitor] no metrics yet at {metrics_file}")
        return None
    issues = evaluate_guardrails(history, window=window)
    last = history[-1]
    shown = {k: v for k, v in last.items() if k != "step"}
    print(f"[monitor] step={last.get('step')} {shown}")
    for issue in issues:
        print(f"  - {issue.level} {issue.code}: {issue.message}")

    level = worst_level(issues)
    if level == "ERROR":
        Path(stop_file).parent.mkdir(parents=True, exist_ok=True)
        Path(stop_file).write_text("stop", encoding="utf-8")
        print(f"[monitor] ERROR -> wrote STOP sentinel at {stop_file}")
        if do_escalate:
            escalate(issues, metrics_file)
    elif level == "WARN" and do_escalate:
        escalate(issues, metrics_file)
    return level


def main() -> None:
    ap = argparse.ArgumentParser(description="Monitor a training run via its metrics.jsonl.")
    ap.add_argument("--metrics-file", default="data/outputs/qwen-chess-logic-grpo/metrics.jsonl")
    ap.add_argument("--stop-file", default=None, help="Sentinel written on ERROR (default: <metrics dir>/STOP)")
    ap.add_argument("--interval", type=int, default=30)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--escalate", action="store_true", help="Spawn a Cursor agent on WARN/ERROR")
    args = ap.parse_args()

    stop_file = args.stop_file or str(Path(args.metrics_file).parent / "STOP")

    if args.once:
        run_once(args.metrics_file, stop_file, args.window, args.escalate)
        return

    print(f"[monitor] watching {args.metrics_file} every {args.interval}s (Ctrl-C to stop)")
    while True:
        try:
            run_once(args.metrics_file, stop_file, args.window, args.escalate)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[monitor] stopped")
            break


if __name__ == "__main__":
    main()

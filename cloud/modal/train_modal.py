"""Modal entrypoints for SFT and GRPO training, with Trackio monitoring.

Phases (run from the project root):

    modal run cloud/modal/train_modal.py::sft  --config /app/configs/training/qwen_lora.yaml
    modal run cloud/modal/train_modal.py::grpo --config /app/configs/training/qwen_grpo.yaml

Metrics stream to a Trackio Space dashboard (when TRACKIO_SPACE_ID is set) and to
``<output_dir>/metrics.jsonl`` on the data volume; the in-process GuardrailCallback
fires alerts and auto-stops on divergence. See docs/RUNBOOK.md for required secrets.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "chess-logic-gpt-train"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch",
        "transformers",
        "datasets",
        "accelerate",
        "peft",
        "trl",
        "vllm",
        "bitsandbytes",
        "trackio",
        "pyyaml",
        "orjson",
        "python-chess",
        "zstandard",
        "tqdm",
    )
    .add_local_file(str(PROJECT_ROOT / "pyproject.toml"), remote_path="/app/pyproject.toml")
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path="/app/scripts")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/app/configs")
    .run_commands("cd /app && pip install -e .")
)

app = modal.App(APP_NAME, image=image)
volume = modal.Volume.from_name("chess-logic-gpt-data", create_if_missing=True)

# huggingface-secret must provide HF_TOKEN (model pull + Trackio Space sync).
SECRETS = [modal.Secret.from_name("huggingface-secret")]
GPU = os.environ.get("CLG_GPU", "H100")


def _run(script: str, config_path: str, trackio_space: str | None, run_name: str) -> None:
    import subprocess

    os.chdir("/app")
    env = dict(os.environ)
    env["TRACKIO_PROJECT"] = "chess-logic-gpt"
    env["TRACKIO_RUN"] = run_name
    if trackio_space:
        env["TRACKIO_SPACE_ID"] = trackio_space
    subprocess.run(["python", script, "--config", config_path], check=True, env=env)
    volume.commit()


@app.function(gpu=GPU, timeout=60 * 60 * 24, volumes={"/app/data": volume}, secrets=SECRETS)
def sft(config_path: str = "/app/configs/training/qwen_lora.yaml", trackio_space: str | None = None) -> None:
    _run("scripts/train_lora.py", config_path, trackio_space, "sft")


@app.function(gpu=GPU, timeout=60 * 60 * 24, volumes={"/app/data": volume}, secrets=SECRETS)
def grpo(config_path: str = "/app/configs/training/qwen_grpo.yaml", trackio_space: str | None = None) -> None:
    _run("scripts/train_grpo.py", config_path, trackio_space, "grpo")


@app.function(gpu=GPU, timeout=60 * 60 * 6, volumes={"/app/data": volume}, secrets=SECRETS)
def evaluate(
    model: str,
    adapter: str | None = None,
    data: str = "/app/data/processed/eval_mix.jsonl",
    out: str = "/app/data/outputs/eval_report.json",
) -> None:
    import subprocess

    os.chdir("/app")
    cmd = ["python", "scripts/evaluate.py", "--model", model, "--data", data, "--out", out]
    if adapter:
        cmd += ["--adapter", adapter]
    subprocess.run(cmd, check=True)
    volume.commit()


# Back-compat alias for the original entrypoint name.
@app.function(gpu=GPU, timeout=60 * 60 * 24, volumes={"/app/data": volume}, secrets=SECRETS)
def train(config_path: str = "/app/configs/training/qwen_lora.yaml") -> None:
    _run("scripts/train_lora.py", config_path, None, "sft")

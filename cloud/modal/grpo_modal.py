"""Modal GRPO entrypoint — RLVR probe off the enriched SFT adapter.

Pulls train_mix + the base/SFT adapter from HF, merges the SFT LoRA into the
base, then runs GRPO (verifiable reward) learning a fresh LoRA on an A100-80GB,
and pushes the resulting adapter back to HF.

    modal run cloud/modal/grpo_modal.py

Requires the Modal secret `huggingface-secret` (HF_TOKEN).
"""

from pathlib import Path

import modal

_here = Path(__file__).resolve()
# parents[2] is the repo root locally; in Modal's container __file__ has no such
# depth — fall back so the module imports cleanly there (path is only used locally).
PROJECT_ROOT = _here.parents[2] if len(_here.parents) >= 3 else Path("/app")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51",
        "trl>=0.12",
        "peft>=0.12",
        "accelerate>=0.33",
        "datasets>=2.20",
        "bitsandbytes",
        "python-chess",
        "huggingface_hub",
        "orjson",
        "pyyaml",
        "trackio",
    )
    .add_local_file(str(PROJECT_ROOT / "pyproject.toml"), remote_path="/app/pyproject.toml")
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path="/app/scripts")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/app/configs")
)

app = modal.App("clg-grpo", image=image)


@app.function(gpu="A100-80GB", timeout=60 * 60 * 4, secrets=[modal.Secret.from_name("huggingface-secret")])
def grpo(
    config_path: str = "configs/training/qwen3_8b_grpo_probe.yaml",
    push_repo: str = "Tobiasd2/chess-logic-gpt-8b-grpo-probe",
) -> None:
    import os
    import shutil
    import subprocess

    from huggingface_hub import HfApi, hf_hub_download

    os.chdir("/app")
    env = {**os.environ, "PYTHONPATH": "/app/src"}
    os.makedirs("data/processed", exist_ok=True)
    tm = hf_hub_download(
        "Tobiasd2/chess-logic-gpt-data",
        "processed/train_mix.jsonl",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    shutil.copy(tm, "data/processed/train_mix.jsonl")

    subprocess.run(["python", "scripts/train_grpo.py", "--config", config_path], check=True, env=env)

    out = "data/outputs/qwen3-8b-grpo-probe"
    if os.path.isdir(out):
        api = HfApi()
        api.create_repo(push_repo, private=True, exist_ok=True, repo_type="model", token=os.environ["HF_TOKEN"])
        api.upload_folder(folder_path=out, repo_id=push_repo, repo_type="model", token=os.environ["HF_TOKEN"])
        print("pushed GRPO adapter ->", push_repo, flush=True)


@app.local_entrypoint()
def main(config_path: str = "configs/training/qwen3_8b_grpo_probe.yaml"):
    grpo.remote(config_path)

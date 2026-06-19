"""Modal eval entrypoint — OOD eval of an HF-hosted adapter, quota-free.

Pulls the base model (public), the LoRA adapter (private HF repo), and the OOD
puzzle set (private HF dataset) and scores with the project verifier on a cheap
A10G. Kaggle's weekly GPU quota doesn't apply here.

    modal run cloud/modal/eval_modal.py --adapter Tobiasd2/chess-logic-gpt-8b-sft-enriched

Requires a Modal secret `huggingface-secret` providing HF_TOKEN.
"""

from pathlib import Path

import modal

_here = Path(__file__).resolve()
# parents[2] is the repo root locally (cloud/modal/eval_modal.py); in Modal's
# container __file__ is /root/eval_modal.py with no such depth — fall back so the
# module imports cleanly there (the path is only used locally to build the image).
PROJECT_ROOT = _here.parents[2] if len(_here.parents) >= 3 else Path("/app")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers>=4.51",
        "peft>=0.12",
        "accelerate>=0.33",
        "datasets>=2.20",
        "bitsandbytes",
        "python-chess",
        "huggingface_hub",
        "orjson",
        "pyyaml",
    )
    .add_local_file(str(PROJECT_ROOT / "pyproject.toml"), remote_path="/app/pyproject.toml")
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/app/src")
    .add_local_dir(str(PROJECT_ROOT / "scripts"), remote_path="/app/scripts")
    .add_local_dir(str(PROJECT_ROOT / "configs"), remote_path="/app/configs")
)

app = modal.App("clg-eval", image=image)


@app.function(gpu="A100-40GB", timeout=60 * 60, secrets=[modal.Secret.from_name("huggingface-secret")])
def evaluate_adapter(adapter: str, limit: int = 300) -> str:
    import os
    import subprocess

    from huggingface_hub import hf_hub_download

    os.chdir("/app")
    env = {**os.environ, "PYTHONPATH": "/app/src"}
    ood = hf_hub_download(
        "Tobiasd2/chess-logic-gpt-data",
        "processed/eval_puzzles_ood.jsonl",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    subprocess.run(
        [
            "python", "scripts/evaluate.py",
            "--model", "Qwen/Qwen3-8B",
            "--adapter", adapter,
            "--data", ood,
            "--out", "/tmp/report.json",
            "--load-in-4bit", "--shuffle", "--seed", "0",
            "--limit", str(limit), "--max-new-tokens", "256", "--debug", "6",
        ],
        check=True,
        env=env,
    )
    report = Path("/tmp/report.json").read_text()
    # Persist to HF so the result survives even if the local client disconnects
    # (we run detached); retrieve with hf_hub_download(reports/eval_ood.json).
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj="/tmp/report.json",
        path_in_repo="reports/eval_ood.json",
        repo_id="Tobiasd2/chess-logic-gpt-data",
        repo_type="dataset",
        token=os.environ["HF_TOKEN"],
    )
    print("==== EVAL REPORT ====", flush=True)
    print(report, flush=True)
    return report


@app.local_entrypoint()
def main(adapter: str = "Tobiasd2/chess-logic-gpt-8b-sft-enriched", limit: int = 300):
    evaluate_adapter.remote(adapter, limit)

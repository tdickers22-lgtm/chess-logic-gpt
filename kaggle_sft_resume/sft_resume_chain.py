#!/usr/bin/env python3
"""Kaggle SFT *resume chain* for chess-logic-gpt (calc-trace data3).

Built on the PROVEN train_kernel.py recipe so it actually runs:
  - pinned torch==2.6.0 + matched torchvision/torchaudio (Kaggle still hands out
    P100 sm_60; the default cu121 build dropped Pascal -> the unpinned install was
    what raised `ModuleNotFoundError: PreTrainedModel` last time),
  - bitsandbytes + the project installed with --no-deps so they can't drag torch,
  - trl omitted (GRPO-only),
  - /kaggle/input discovery (no in-kernel `kaggle datasets download`; the data/src
    are auto-mounted because they're declared as dataset_sources),
  - runs the REAL configs/training YAML (smoke fail-fast, then qwen3_8b_lora).

Cross-session chaining: before the 8B run it copies the latest checkpoint from the
attached chess-logic-gpt-checkpoints dataset into the output dir, so HF Trainer's
get_last_checkpoint (resume_from_checkpoint: true in the config) picks up where the
previous session stopped. After the run it leaves the new checkpoint in
/kaggle/working (always retrievable via `kaggle kernels output`) and best-effort
pushes it back to the checkpoints dataset for the next session.

Run once per free Kaggle session to accumulate epochs on the 8B QLoRA SFT.
"""

import glob
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-logic-gpt"
CKPT_DATASET = "tobiasdicker/chess-logic-gpt-checkpoints"
# Must match training.output_dir in configs/training/qwen3_8b_lora.yaml (relative to REPO).
OUT_SUBDIR = "data/outputs/qwen3-8b-chess-logic-lora"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# 0. Show what's actually attached (self-diagnosing log).
roots = [Path(p) for p in sorted(glob.glob("/kaggle/input/*"))]
print("INPUT roots:", roots, flush=True)
for r in roots:
    for p in sorted(r.rglob("*"))[:40]:
        print("   ", p, flush=True)


def locate(name):
    for r in roots:
        hits = sorted(r.rglob(name))
        if hits:
            return hits[0]
    return None


# 1. Stage the source: prefer the tarball, fall back to an extracted tree.
if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
tar = locate("src.tar.gz")
pyproj = locate("pyproject.toml")
if tar:
    print("extracting", tar, flush=True)
    with tarfile.open(tar) as t:
        t.extractall(REPO)
elif pyproj:
    print("copying source tree from", pyproj.parent, flush=True)
    shutil.copytree(pyproj.parent, REPO, dirs_exist_ok=True)
else:
    raise SystemExit("project source not found under /kaggle/input")
os.chdir(REPO)

# 2. PROVEN install recipe (pinned; see module docstring for why).
sh([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
    "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
    "transformers>=4.51", "peft>=0.12", "accelerate>=0.33",
    "datasets>=2.20", "trackio", "python-chess", "orjson", "pyyaml", "zstandard"])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "bitsandbytes"])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."])

# 3. Wire the prepared calc-trace JSONL (data3) into data/processed/.
proc = REPO / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)
for name in ("train_mix.jsonl", "eval_mix.jsonl", "eval_puzzles_ood.jsonl"):
    f = locate(name)
    if f and not (proc / name).exists():
        os.symlink(f, proc / name)
print("data:", sorted(p.name for p in proc.glob("*.jsonl")), flush=True)

# 4. Pull the latest prior checkpoint into the output dir so HF Trainer's
#    get_last_checkpoint (resume_from_checkpoint: true) resumes it. First session
#    finds nothing -> fresh start that bootstraps the chain.
out_dir = REPO / OUT_SUBDIR
out_dir.mkdir(parents=True, exist_ok=True)
prior = None
for r in roots:
    hits = [h for h in sorted(r.rglob("checkpoint-*")) if h.is_dir()]
    if hits:
        # numeric sort by step so checkpoint-400 beats checkpoint-200
        hits.sort(key=lambda p: int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else -1)
        prior = hits[-1]
        break
if prior:
    dest = out_dir / prior.name
    print(f"RESUMING from {prior} -> {dest}", flush=True)
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(prior, dest)
else:
    print("no prior checkpoint under /kaggle/input -> FRESH start (bootstrapping the chain)", flush=True)

env = dict(os.environ, TRACKIO_PROJECT="chess-logic-gpt", TOKENIZERS_PARALLELISM="false")

# 5. Smoke (0.5B, 20 steps) fail-fast, then the real Qwen3-8B QLoRA SFT (resumes if a
#    checkpoint was staged in step 4).
sh([sys.executable, "scripts/train_lora.py", "--config", "configs/training/smoke.yaml"], env=env)
sh([sys.executable, "scripts/train_lora.py", "--config", "configs/training/qwen3_8b_lora.yaml"], env=env)

# 6. Persist outputs to /kaggle/working (always retrievable via `kaggle kernels output`).
for out in ("data/outputs/smoke", OUT_SUBDIR):
    src = REPO / out
    if src.exists():
        shutil.copytree(src, WORK / Path(out).name, dirs_exist_ok=True)

# 7. Best-effort push of the latest 8B checkpoint back to the checkpoints dataset so the
#    NEXT session resumes automatically. If in-kernel Kaggle auth isn't set, this fails
#    gracefully and the checkpoint is still in /kaggle/working for a local pull+publish.
try:
    from transformers.trainer_utils import get_last_checkpoint

    latest = get_last_checkpoint(str(out_dir)) if out_dir.is_dir() else None
    if latest:
        print("publishing checkpoint:", latest, flush=True)
        up = WORK / "ckpt_upload"
        if up.exists():
            shutil.rmtree(up)
        up.mkdir(parents=True)
        name = Path(latest).name
        shutil.copytree(latest, up / name)
        (up / "dataset-metadata.json").write_text(json.dumps({
            "title": "chess-logic-gpt-checkpoints",
            "id": CKPT_DATASET,
            "licenses": [{"name": "CC0-1.0"}],
        }))
        # Dataset already exists (README) -> version it.
        r = subprocess.run(
            ["kaggle", "datasets", "version", "-p", str(up), "-m", f"SFT {name}", "--dir-mode", "zip"],
            capture_output=True, text=True,
        )
        print(r.stdout, r.stderr, flush=True)
        if r.returncode == 0:
            print("checkpoint published to", CKPT_DATASET, flush=True)
        else:
            print("in-kernel push failed (expected if no Kaggle secret); pull from /kaggle/working instead", flush=True)
    else:
        print("no checkpoint produced", flush=True)
except Exception as e:
    print(f"checkpoint publish step errored (non-fatal): {e}", flush=True)

print("DONE", flush=True)

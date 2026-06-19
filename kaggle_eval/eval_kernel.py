#!/usr/bin/env python3
"""Kaggle eval kernel: OOD chess-puzzle eval of the partial 8B SFT adapter.

Inputs (attached datasets, discovered dynamically under /kaggle/input):
  - chess-logic-gpt-data    -> src.tar.gz (project source) + eval_puzzles_ood.jsonl
  - chess-logic-gpt-adapter -> adapter_config.json + adapter_model.safetensors (checkpoint-400)
Base Qwen3-8B is pulled public (internet on), loaded in 4-bit so it fits a 16GB
GPU, with the LoRA adapter applied. Scores with the project's verifier
(earned move accuracy). Report -> /kaggle/working/eval_report_ood.json.
"""

import glob
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
REPO = WORK / "chess-logic-gpt"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


roots = [Path(p) for p in sorted(glob.glob("/kaggle/input/*"))]
print("INPUT roots:", roots, flush=True)


def locate(name):
    for r in roots:
        hits = sorted(r.rglob(name))
        if hits:
            return hits[0]
    return None


# 1. Stage source from the DEDICATED src dataset (deterministic — the data
#    dataset also carries a bundled source tree, which may be stale; never use it
#    for code). Falls back to a general scan only if no src dataset is attached.
def pick(name, prefer="chess-logic-gpt-src"):
    """Find `name` under any input, preferring a path in the dedicated src
    dataset (robust to however Kaggle nests the mount)."""
    hits = []
    for r in roots:
        hits += sorted(r.rglob(name))
    chosen = [h for h in hits if prefer in str(h)] or hits
    return chosen[0] if chosen else None


if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
tar = pick("src.tar.gz")
pyproj = pick("pyproject.toml")
if tar:
    with tarfile.open(tar) as t:
        t.extractall(REPO)
elif pyproj:
    shutil.copytree(pyproj.parent, REPO, dirs_exist_ok=True)
else:
    raise SystemExit("project source not found in src dataset")
os.chdir(REPO)
sh([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
    "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
    "transformers>=4.51", "peft>=0.12", "accelerate>=0.33",
    "datasets>=2.20", "trackio", "python-chess", "orjson", "pyyaml", "zstandard"])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "bitsandbytes"])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."])

# 2. Locate the OOD eval set and the adapter.
ood = locate("eval_puzzles_ood.jsonl")
adapter_cfg = locate("adapter_config.json")
adapter_dir = adapter_cfg.parent if adapter_cfg else None
print("ood:", ood, "| adapter_dir:", adapter_dir, flush=True)
if not ood or not adapter_dir:
    raise SystemExit("missing OOD data or adapter under /kaggle/input")

# 3. Eval: base in 4-bit (fits 16GB) + LoRA adapter, shuffled 300-record sample.
out = str(WORK / "eval_report_ood.json")
sh([sys.executable, "scripts/evaluate.py",
    "--model", "Qwen/Qwen3-8B",
    "--adapter", str(adapter_dir),
    "--data", str(ood),
    "--out", out,
    "--load-in-4bit", "--shuffle", "--seed", "0",
    "--limit", "300", "--debug", "5", "--max-new-tokens", "256"])
print("EVAL DONE ->", out, flush=True)

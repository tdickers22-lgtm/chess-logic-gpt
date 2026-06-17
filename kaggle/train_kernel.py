#!/usr/bin/env python3
"""Kaggle script kernel: SFT for chess-logic-gpt on a free 2xT4.

Reads the project source (src.tar.gz, or an auto-extracted tree) and the
prepared JSONL from the attached private Kaggle dataset -- located dynamically
under /kaggle/input so it works regardless of the exact mount slug or whether
Kaggle extracted the archive. Base models (Qwen2.5-0.5B smoke, Qwen3-8B) are
public, so only enable_internet is needed -- no token lives on Kaggle.
Validates with a 20-step 0.5B smoke, then runs the Qwen3-8B QLoRA SFT.
Outputs land in /kaggle/working.
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


# 0. Show what's actually attached (so the log is self-diagnosing).
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
sh([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "trl>=0.12", "peft>=0.12", "accelerate>=0.33",
    "datasets>=2.20", "bitsandbytes", "trackio", "python-chess", "orjson",
    "pyyaml", "zstandard"])
sh([sys.executable, "-m", "pip", "install", "-q", "-e", "."])

# 2. Wire the prepared JSONL into data/processed/.
proc = REPO / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)
for name in ("train_mix.jsonl", "eval_mix.jsonl", "eval_puzzles_ood.jsonl"):
    f = locate(name)
    if f and not (proc / name).exists():
        os.symlink(f, proc / name)
print("data:", sorted(p.name for p in proc.glob("*.jsonl")), flush=True)

env = dict(os.environ, TRACKIO_PROJECT="chess-logic-gpt", TOKENIZERS_PARALLELISM="false")

# 3. Smoke (0.5B, 20 steps) -> fail fast, then the real Qwen3-8B QLoRA SFT.
sh([sys.executable, "scripts/train_lora.py", "--config", "configs/training/smoke.yaml"], env=env)
sh([sys.executable, "scripts/train_lora.py", "--config", "configs/training/qwen3_8b_lora.yaml"], env=env)

# 4. Persist outputs to /kaggle/working.
for out in ("data/outputs/smoke", "data/outputs/qwen3-8b-chess-logic-lora"):
    src = REPO / out
    if src.exists():
        shutil.copytree(src, WORK / Path(out).name, dirs_exist_ok=True)
print("DONE", flush=True)

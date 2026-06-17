#!/usr/bin/env python3
"""Kaggle script kernel: SFT for chess-logic-gpt on a free 2xT4.

Everything comes from one attached private Kaggle dataset:
  /kaggle/input/chess-logic-gpt-data/src.tar.gz  -> project source
  /kaggle/input/chess-logic-gpt-data/*.jsonl      -> prepared train/eval data
The base models (Qwen2.5-0.5B smoke, Qwen3-8B) are public, so only
enable_internet is needed -- no HF/GitHub token lives on Kaggle. Validates the
path with a 20-step 0.5B smoke, then runs the Qwen3-8B QLoRA SFT. Outputs land
in /kaggle/working (kernel output, downloadable / reusable as input to resume).
"""

import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

WORK = Path("/kaggle/working")
DATA_IN = Path("/kaggle/input/chess-logic-gpt-data")
REPO = WORK / "chess-logic-gpt"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# 1. Unpack the source and install the package.
if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
with tarfile.open(DATA_IN / "src.tar.gz") as t:
    t.extractall(REPO)
os.chdir(REPO)
sh([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "trl>=0.12", "peft>=0.12", "accelerate>=0.33",
    "datasets>=2.20", "bitsandbytes", "trackio", "python-chess", "orjson",
    "pyyaml", "zstandard"])
sh([sys.executable, "-m", "pip", "install", "-q", "-e", "."])

# 2. Wire the prepared JSONL into data/processed/.
proc = REPO / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)
for f in DATA_IN.glob("*.jsonl"):
    dst = proc / f.name
    if not dst.exists():
        os.symlink(f, dst)
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

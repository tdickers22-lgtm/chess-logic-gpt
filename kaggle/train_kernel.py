#!/usr/bin/env python3
"""Kaggle script kernel: SFT for chess-logic-gpt on a free 2xT4.

Reads the project source and the prepared JSONL from two attached Kaggle
datasets (so no HF/GitHub token needs to live on Kaggle), validates the path
with a 20-step 0.5B smoke, then runs the Qwen3-8B QLoRA SFT with frequent
checkpoints. The base model is public, so only `enable_internet` is required.

Attach as dataset_sources (see kernel-metadata.json):
  - <user>/chess-logic-gpt-src   -> /kaggle/input/chess-logic-gpt-src
  - <user>/chess-logic-gpt-data  -> /kaggle/input/chess-logic-gpt-data
Outputs (adapter + metrics) land in /kaggle/working and persist as kernel output.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working")
SRC_IN = Path("/kaggle/input/chess-logic-gpt-src")
DATA_IN = Path("/kaggle/input/chess-logic-gpt-data")
REPO = WORK / "chess-logic-gpt"


def sh(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


# 1. Stage source into a writable dir and install the package.
if REPO.exists():
    shutil.rmtree(REPO)
shutil.copytree(SRC_IN, REPO)
os.chdir(REPO)
sh([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.45", "trl>=0.12", "peft>=0.12", "accelerate>=0.33",
    "datasets>=2.20", "bitsandbytes", "trackio", "python-chess", "orjson",
    "pyyaml", "zstandard"])
sh([sys.executable, "-m", "pip", "install", "-q", "-e", "."])

# 2. Wire the attached JSONL into data/processed/ (symlink; read-only is fine).
proc = REPO / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)
src_data = (DATA_IN / "processed") if (DATA_IN / "processed").is_dir() else DATA_IN
for f in src_data.glob("*.jsonl"):
    dst = proc / f.name
    if not dst.exists():
        os.symlink(f, dst)
print("data files:", sorted(p.name for p in proc.glob("*.jsonl")), flush=True)

env = dict(os.environ, TRACKIO_PROJECT="chess-logic-gpt", TOKENIZERS_PARALLELISM="false")

# 3. Smoke (0.5B, 20 steps) -> fail fast on any path bug before the real run.
sh([sys.executable, "scripts/train_lora.py", "--config", "configs/training/smoke.yaml"], env=env)

# 4. The real SFT: Qwen3-8B QLoRA, checkpointed for resume across sessions.
sh([sys.executable, "scripts/train_lora.py", "--config", "configs/training/qwen3_8b_lora.yaml"], env=env)

# 5. Persist outputs to /kaggle/working (kernel output, downloadable / reusable).
for out in ("data/outputs/smoke", "data/outputs/qwen3-8b-chess-logic-lora"):
    src = REPO / out
    if src.exists():
        shutil.copytree(src, WORK / Path(out).name, dirs_exist_ok=True)
print("DONE", flush=True)

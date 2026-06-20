#!/usr/bin/env python3
"""Kaggle GRPO probe: RLVR on top of the enriched SFT adapter, on a free 16GB GPU.

Reads everything from attached private Kaggle datasets (no HF/GitHub token):
  chess-logic-gpt-src             -> src.tar.gz (project source)
  chess-logic-gpt-data2           -> train_mix.jsonl
  chess-logic-gpt-adapter-enriched-> adapter_config.json + adapter_model.safetensors
The base Qwen3-8B is public (internet on). 4-bit base + continue-the-adapter so
it fits 16GB. Output adapter -> /kaggle/working/qwen3-8b-grpo-kaggle.
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


def pick(name, prefer=None):
    hits = []
    for r in roots:
        hits += sorted(r.rglob(name))
    if prefer:
        pref = [h for h in hits if prefer in str(h)]
        if pref:
            return pref[0]
    return hits[0] if hits else None


# 1. Source (from the dedicated src dataset).
if REPO.exists():
    shutil.rmtree(REPO)
REPO.mkdir(parents=True)
tar = pick("src.tar.gz", "chess-logic-gpt-src") or pick("src.tar.gz")
pyproj = pick("pyproject.toml", "chess-logic-gpt-src")
if tar:
    with tarfile.open(tar) as t:
        t.extractall(REPO)
elif pyproj:
    shutil.copytree(pyproj.parent, REPO, dirs_exist_ok=True)
else:
    raise SystemExit("project source not found")
os.chdir(REPO)

# 2. Deps (torch 2.6 trio for P100/T4 compat; trl for GRPO; bitsandbytes no-deps).
sh([sys.executable, "-m", "pip", "install", "-q", "--upgrade-strategy", "only-if-needed",
    "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0",
    "transformers>=4.51", "trl>=0.12", "peft>=0.12", "accelerate>=0.33",
    "datasets>=2.20", "trackio", "python-chess", "orjson", "pyyaml", "zstandard"])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "bitsandbytes"])
sh([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", "."])

# 3. Wire data + the local enriched adapter.
proc = REPO / "data" / "processed"
proc.mkdir(parents=True, exist_ok=True)
tm = pick("train_mix.jsonl")
if tm and not (proc / "train_mix.jsonl").exists():
    os.symlink(tm, proc / "train_mix.jsonl")
adapter_cfg = pick("adapter_config.json", "chess-logic-gpt-adapter")
adapter_dir = str(adapter_cfg.parent) if adapter_cfg else ""
print("train_mix:", tm, "| adapter_dir:", adapter_dir, flush=True)

env = dict(os.environ, TRACKIO_PROJECT="chess-logic-gpt", TOKENIZERS_PARALLELISM="false")
if adapter_dir:
    env["CLG_SFT_ADAPTER"] = adapter_dir

# 4. GRPO probe.
sh([sys.executable, "scripts/train_grpo.py", "--config", "configs/training/qwen3_8b_grpo_kaggle.yaml"], env=env)

# 5. Persist the GRPO adapter.
out = REPO / "data" / "outputs" / "qwen3-8b-grpo-kaggle"
if out.exists():
    shutil.copytree(out, WORK / "qwen3-8b-grpo-kaggle", dirs_exist_ok=True)
print("DONE", flush=True)

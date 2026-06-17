#!/usr/bin/env python3
"""Pull prepared JSONL datasets from a Hugging Face dataset repo to local disk.

The cloud-agnostic counterpart to `push_data_to_hub.py`: on any GPU box, this is
all that's needed to stage training data (just HF_TOKEN + the repo id).

    HF_TOKEN=hf_xxx python scripts/fetch_data.py --repo Tobiasd2/chess-logic-gpt-data
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

DEFAULT_FILES = ["train_mix.jsonl", "eval_mix.jsonl", "eval_puzzles_ood.jsonl"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. Tobiasd2/chess-logic-gpt-data")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.files:
        cached = hf_hub_download(
            repo_id=args.repo,
            repo_type="dataset",
            filename=f"processed/{name}",
            token=token,
        )
        dest = out_dir / name
        shutil.copyfile(cached, dest)
        print(f"fetched {name} -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")

    print("data staged.")


if __name__ == "__main__":
    main()

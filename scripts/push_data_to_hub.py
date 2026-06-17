#!/usr/bin/env python3
"""Push prepared JSONL datasets to a (private) Hugging Face dataset repo.

This makes the data portable: any GPU box (Modal / Thunder / RunPod / Azure)
pulls it with `scripts/fetch_data.py` using only HF_TOKEN. Re-run after
regenerating data to refresh the repo.

    HF_TOKEN=hf_xxx python scripts/push_data_to_hub.py --repo Tobiasd2/chess-logic-gpt-data
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_FILES = [
    "data/processed/train_mix.jsonl",
    "data/processed/eval_mix.jsonl",
    "data/processed/eval_puzzles_ood.jsonl",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="e.g. Tobiasd2/chess-logic-gpt-data")
    ap.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    ap.add_argument("--public", action="store_true", help="Make the dataset repo public (default: private)")
    args = ap.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("set HF_TOKEN in the environment")

    api = HfApi(token=token)
    api.create_repo(args.repo, repo_type="dataset", private=not args.public, exist_ok=True)
    print(f"repo ready: https://huggingface.co/datasets/{args.repo} (private={not args.public})")

    for local in args.files:
        path = Path(local)
        if not path.exists():
            print(f"  skip (missing): {local}")
            continue
        size_mb = path.stat().st_size / 1e6
        dest = f"processed/{path.name}"
        print(f"  uploading {local} -> {dest} ({size_mb:.1f} MB) ...", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=dest,
            repo_id=args.repo,
            repo_type="dataset",
        )
        print(f"  done: {dest}", flush=True)

    print(f"all uploads complete -> https://huggingface.co/datasets/{args.repo}")


if __name__ == "__main__":
    main()

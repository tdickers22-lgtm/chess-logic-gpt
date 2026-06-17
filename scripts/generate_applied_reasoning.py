#!/usr/bin/env python3
from __future__ import annotations

import argparse

from chess_logic_gpt.logic.applied import generate_applied_reasoning_examples
from chess_logic_gpt.records import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Generate clean synthetic applied reasoning examples across philosophy and social domains."
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write_jsonl(args.out, generate_applied_reasoning_examples(args.n, args.seed))
    print(f"wrote {args.n} applied reasoning records to {args.out}")


if __name__ == "__main__":
    main()

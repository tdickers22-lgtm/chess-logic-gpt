#!/usr/bin/env python3
from __future__ import annotations

import argparse

from chess_logic_gpt.logic.generate import generate_logic_examples
from chess_logic_gpt.records import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate clean synthetic Fitch/predicate logic examples.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write_jsonl(args.out, generate_logic_examples(args.n, args.seed))
    print(f"wrote {args.n} logic records to {args.out}")


if __name__ == "__main__":
    main()


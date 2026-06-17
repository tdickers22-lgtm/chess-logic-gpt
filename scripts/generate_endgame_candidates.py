#!/usr/bin/env python3
from __future__ import annotations

import argparse

from chess_logic_gpt.chess.endgames import generate_endgame_candidates
from chess_logic_gpt.records import write_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate legal low-piece endgame candidates for Syzygy labeling.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rows = generate_endgame_candidates(args.n, args.seed)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} endgame candidate records to {args.out}")


if __name__ == "__main__":
    main()

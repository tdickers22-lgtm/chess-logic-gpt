#!/usr/bin/env python3
"""Build an out-of-distribution chess puzzle eval set.

The held-out set is intentionally *harder* than training (higher rating band) and
its FENs are deduped against the training files, so accuracy here measures motif
*generalization* rather than memorization of drilled positions.

Example
-------
    python scripts/build_ood_eval.py \
        --infile data/raw/lichess_db_puzzle.csv.zst \
        --train data/processed/lichess_puzzles.jsonl \
        --out data/processed/eval_puzzles_ood.jsonl \
        --min-rating 2300 --max-rating 2900 --per-motif 300
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from chess_logic_gpt.chess.curriculum import DEFAULT_CORE_MOTIFS, motif_counts
from chess_logic_gpt.chess.puzzles import harvest_motif_pool
from chess_logic_gpt.records import write_jsonl


def load_train_fens(paths: list[str]) -> set[str]:
    fens: set[str] = set()
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                fen = json.loads(line).get("metadata", {}).get("fen")
                if fen:
                    fens.add(fen)
    return fens


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infile", required=True, help="Lichess puzzle CSV(.zst)")
    ap.add_argument("--train", nargs="+", required=True, help="Training JSONL files whose FENs to exclude")
    ap.add_argument("--out", default="data/processed/eval_puzzles_ood.jsonl")
    ap.add_argument("--min-rating", type=int, default=2300)
    ap.add_argument("--max-rating", type=int, default=2900)
    ap.add_argument("--per-motif", type=int, default=300)
    ap.add_argument("--cap-factor", type=float, default=2.0, help="Harvest per_motif*cap_factor before dedupe")
    ap.add_argument("--motifs", default="", help="Comma-separated motifs (default: core motifs)")
    args = ap.parse_args()

    motifs = args.motifs.split(",") if args.motifs else DEFAULT_CORE_MOTIFS
    cap = int(args.per_motif * args.cap_factor) + 1

    print(f"loading training FENs from {len(args.train)} file(s) ...")
    train_fens = load_train_fens(args.train)
    print(f"  {len(train_fens)} distinct training FENs to exclude")

    print(f"harvesting OOD band [{args.min_rating}, {args.max_rating}] from {args.infile} ...")
    pool = harvest_motif_pool(
        args.infile,
        motifs=motifs,
        cap_distinct=cap,
        min_rating=args.min_rating,
        max_rating=args.max_rating,
    )

    kept: list[dict] = []
    per_motif: collections.Counter[str] = collections.Counter()
    leaked = 0
    for record in pool:
        fen = record["metadata"]["fen"]
        motif = record["metadata"]["primary_motif"]
        if fen in train_fens:
            leaked += 1
            continue
        if per_motif[motif] >= args.per_motif:
            continue
        per_motif[motif] += 1
        kept.append(record)

    print(f"excluded {leaked} positions that overlapped training")
    print("OOD eval positions per motif:")
    for motif, count in sorted(motif_counts(kept).items(), key=lambda kv: -kv[1]):
        print(f"  {motif:>20}: {count}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, kept)
    print(f"wrote {len(kept)} OOD eval records to {args.out}")


if __name__ == "__main__":
    main()

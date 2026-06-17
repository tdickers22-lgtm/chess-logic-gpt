#!/usr/bin/env python3
"""Ingest the Lichess open puzzle database (CC0) into motif-drilled chess records.

Examples
--------
Download + ingest a motif-balanced drilling set (Woodpecker-style repetition):

    python scripts/ingest_puzzles.py --download \
        --out data/processed/chess_puzzles.jsonl \
        --limit 400000 --min-rating 800 --max-rating 2200 \
        --curriculum --per-motif 1500 --repeat 3 --order blocked_then_interleaved

Ingest from an already-downloaded file (plain .csv or .csv.zst):

    python scripts/ingest_puzzles.py --infile data/raw/lichess_db_puzzle.csv.zst \
        --out data/processed/chess_puzzles.jsonl --limit 200000
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from chess_logic_gpt.chess.curriculum import (
    DEFAULT_CORE_MOTIFS,
    build_motif_curriculum,
    motif_counts,
)
from chess_logic_gpt.chess.puzzles import (
    LICHESS_PUZZLE_URL,
    harvest_motif_pool,
    puzzles_from_csv,
)
from chess_logic_gpt.records import write_jsonl

DEFAULT_RAW = Path("data/raw/lichess_db_puzzle.csv.zst")


def download(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"already downloaded: {dest}")
        return dest
    print(f"downloading {LICHESS_PUZZLE_URL} -> {dest} (this is a few hundred MB)...")
    urllib.request.urlretrieve(LICHESS_PUZZLE_URL, dest)  # noqa: S310
    print("done.")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest Lichess puzzles into training records.")
    ap.add_argument("--infile", default=None, help="Path to lichess_db_puzzle.csv[.zst]")
    ap.add_argument("--download", action="store_true", help="Download the DB if --infile is absent")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=None, help="Max rows to read from the CSV")
    ap.add_argument("--min-rating", type=int, default=None)
    ap.add_argument("--max-rating", type=int, default=None)
    ap.add_argument("--curriculum", action="store_true", help="Apply motif-weighted drilling order")
    ap.add_argument("--motifs", default=None, help="Comma-separated motif allow-list for the curriculum")
    ap.add_argument("--per-motif", type=int, default=1500)
    ap.add_argument("--cap-factor", type=float, default=1.5, help="Harvest per_motif*cap_factor distinct positions")
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument(
        "--order",
        default="blocked_then_interleaved",
        choices=["blocked", "interleaved", "blocked_then_interleaved"],
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.infile:
        infile = Path(args.infile)
    elif args.download:
        infile = download(DEFAULT_RAW)
    else:
        raise SystemExit("Provide --infile PATH or --download.")
    if not infile.exists():
        raise SystemExit(f"input file not found: {infile}")

    motifs = args.motifs.split(",") if args.motifs else DEFAULT_CORE_MOTIFS

    if args.curriculum:
        cap = int(args.per_motif * args.cap_factor) + 1
        print(f"harvesting puzzles from {infile} (cap {cap} distinct/motif) ...")
        records = harvest_motif_pool(
            infile,
            motifs=motifs,
            cap_distinct=cap,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
        )
    else:
        print(f"parsing puzzles from {infile} ...")
        records = puzzles_from_csv(
            infile,
            limit=args.limit,
            min_rating=args.min_rating,
            max_rating=args.max_rating,
        )
    print(f"parsed {len(records)} valid puzzle records")

    counts = motif_counts(records)
    print("distinct positions per motif:")
    for motif, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {motif:>20}: {count}")

    if args.curriculum:
        records = build_motif_curriculum(
            records,
            motifs=motifs,
            per_motif=args.per_motif,
            repeat=args.repeat,
            order=args.order,
            seed=args.seed,
        )
        print(f"curriculum sequence length: {len(records)} (order={args.order}, repeat={args.repeat})")

    write_jsonl(args.out, records)
    print(f"wrote {len(records)} records to {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from chess_logic_gpt.records import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Mix domain JSONL files into a 50/25/25 training set.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None, help="Override train output path")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    rng = random.Random(int(cfg.get("seed", 42)))
    by_domain: dict[str, list[dict]] = {}
    for domain, paths in cfg["inputs"].items():
        rows: list[dict] = []
        for raw_path in paths or []:
            path = Path(raw_path)
            if not path.exists():
                continue
            rows.extend(read_jsonl(path))
        rng.shuffle(rows)
        by_domain[domain] = rows

    mix = cfg["target_mix"]
    non_empty = {d: rows for d, rows in by_domain.items() if rows}
    if not non_empty:
        raise SystemExit("No input rows found. Generate or ingest datasets first.")

    max_records = cfg.get("max_records")
    if max_records is None:
        # Use the largest balanced set possible without oversampling.
        max_records = min(int(len(non_empty[d]) / mix[d]) for d in non_empty if mix.get(d, 0) > 0)
    max_records = int(max_records)

    mixed: list[dict] = []
    for domain, ratio in mix.items():
        rows = by_domain.get(domain, [])
        if not rows:
            print(f"warning: no rows for domain {domain}")
            continue
        take = min(len(rows), int(round(max_records * float(ratio))))
        mixed.extend(rows[:take])
        print(f"{domain}: taking {take} / {len(rows)}")
    rng.shuffle(mixed)

    eval_fraction = float(cfg.get("eval_fraction", 0.02))
    eval_n = max(1, int(len(mixed) * eval_fraction)) if len(mixed) > 20 else 0
    eval_rows = mixed[:eval_n]
    train_rows = mixed[eval_n:]

    train_out = Path(args.out or cfg["output"]["train"])
    eval_out = Path(cfg["output"]["eval"])
    write_jsonl(train_out, train_rows)
    write_jsonl(eval_out, eval_rows)
    print(f"wrote {len(train_rows)} train rows to {train_out}")
    print(f"wrote {len(eval_rows)} eval rows to {eval_out}")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()


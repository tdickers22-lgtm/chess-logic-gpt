#!/usr/bin/env python3
"""Tiny CPU verification: confirm the attached data dataset has the ENRICHED
chess reasoning (per-puzzle line narration) vs the old fixed template."""
import glob
import json
from pathlib import Path

roots = [Path(p) for p in glob.glob("/kaggle/input/*")]


def find(name):
    for r in roots:
        hits = sorted(r.rglob(name))
        if hits:
            return hits[0]
    return None


f = find("train_mix.jsonl")
print("train_mix path:", f, flush=True)
n = 0
chess_sample = None
for line in open(f):
    r = json.loads(line)
    n += 1
    if r["domain"] == "chess" and chess_sample is None:
        chess_sample = r["messages"][-1]["content"]
        break
print("FIRST CHESS REASONING:", flush=True)
print(chess_sample, flush=True)
enriched = chess_sample and "key move is" in chess_sample
print("VERDICT:", "ENRICHED" if enriched else "OLD/TEMPLATE", flush=True)

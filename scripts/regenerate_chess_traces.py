#!/usr/bin/env python3
"""Rewrite chess puzzle reasoning traces into engine-grounded *calculation* traces.

The frontier eval proved the SFT model parrots: the old trace stated "the key move
is X" up front, so the model never had to read the board. This regenerates every
chess record's reasoning with `calculation_trace`, backed by a real Stockfish
multipv evaluation, so the move is the *conclusion* of board-specific calculation
(forcing-move search + candidate evals + the verified forced line). The gold answer
(the Lichess solution line) is unchanged -- only the reasoning that teaches *how* to
reach it.

Runs N Stockfish engines in parallel (one per worker). ~15-20 min for 72k on 8 cores.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path

import chess
import chess.engine

from chess_logic_gpt.chess.puzzles import calculation_trace
from chess_logic_gpt.records import read_jsonl, write_jsonl
from chess_logic_gpt.training.trace import wrap_trace

_ENGINE: chess.engine.SimpleEngine | None = None
_DEPTH = 14
_MULTIPV = 3
_STOCKFISH = "stockfish"


def _fmt(score: chess.engine.PovScore) -> str:
    rel = score.relative
    if rel.is_mate():
        n = abs(rel.mate())
        return f"mate in {n}" if rel.mate() > 0 else f"getting mated in {n}"
    return f"{rel.score() / 100:+.1f}"


def _init(stockfish: str, depth: int, multipv: int) -> None:
    global _ENGINE, _DEPTH, _MULTIPV
    _DEPTH, _MULTIPV = depth, multipv
    _ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish)
    _ENGINE.configure({"Threads": 1, "Hash": 64})


def _top_moves(fen: str) -> list[tuple[str, str, int]] | None:
    try:
        board = chess.Board(fen)
        info = _ENGINE.analyse(board, chess.engine.Limit(depth=_DEPTH), multipv=_MULTIPV)
        out = []
        for rank, item in enumerate(info):
            pv = item.get("pv")
            if not pv:
                continue
            out.append((board.san(pv[0]), _fmt(item["score"]), rank))
        return out or None
    except Exception:
        return None


def _process(record: dict) -> str:
    md = record["metadata"]
    fen, line, motif = md["fen"], md["line_uci"], md["primary_motif"]
    side = "White" if chess.Board(fen).turn == chess.WHITE else "Black"
    try:
        top = _top_moves(fen)
        reasoning, answer = calculation_trace(fen, line, motif, side, top_moves=top)
        record["messages"][-1]["content"] = wrap_trace(reasoning, answer)
    except Exception:
        pass  # keep the original trace if anything goes wrong on this row
    import json

    return json.dumps(record, ensure_ascii=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--infile", default="data/processed/lichess_puzzles.jsonl")
    ap.add_argument("--out", default="data/processed/lichess_puzzles.jsonl")
    ap.add_argument("--backup", default="data/processed/lichess_puzzles.templated.jsonl")
    ap.add_argument("--workers", type=int, default=min(8, mp.cpu_count()))
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=3)
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--limit", type=int, default=None, help="process only N records (smoke test)")
    args = ap.parse_args()

    records = list(read_jsonl(args.infile))
    if args.limit:
        records = records[: args.limit]
    total = len(records)
    print(f"regenerating {total} chess traces with {args.workers} Stockfish workers (depth {args.depth}) ...")

    if args.backup and args.out == args.infile and not Path(args.backup).exists():
        write_jsonl(args.backup, list(read_jsonl(args.infile)))
        print(f"backed up original templated traces -> {args.backup}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with mp.Pool(args.workers, initializer=_init, initargs=(args.stockfish, args.depth, args.multipv)) as pool, \
            out_path.open("w", encoding="utf-8") as fh:
        for line in pool.imap_unordered(_process, records, chunksize=32):
            fh.write(line + "\n")
            done += 1
            if done % 5000 == 0 or done == total:
                print(f"  {done}/{total}", flush=True)
    print(f"wrote {done} engine-grounded chess records -> {out_path}")


if __name__ == "__main__":
    main()

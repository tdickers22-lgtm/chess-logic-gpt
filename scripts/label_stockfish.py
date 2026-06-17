#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import chess
import chess.engine

from chess_logic_gpt.records import read_jsonl


def pv_to_san(board: chess.Board, pv: list[chess.Move], max_moves: int = 8) -> list[str]:
    line_board = board.copy(stack=False)
    sans: list[str] = []
    for move in pv[:max_moves]:
        if move not in line_board.legal_moves:
            break
        sans.append(line_board.san(move))
        line_board.push(move)
    return sans


def append_assistant_section(row: dict, section: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            msg["content"] = str(msg.get("content", "")).rstrip() + "\n\n" + section
            return


def main() -> None:
    ap = argparse.ArgumentParser(description="Add Stockfish labels to chess JSONL records.")
    ap.add_argument("--infile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stockfish", default=os.environ.get("STOCKFISH_PATH"))
    ap.add_argument("--depth", type=int, default=14)
    ap.add_argument("--multipv", type=int, default=3)
    args = ap.parse_args()

    if not args.stockfish:
        raise SystemExit("Set --stockfish or STOCKFISH_PATH.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with chess.engine.SimpleEngine.popen_uci(args.stockfish) as engine, out.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.infile):
            fen = row.get("metadata", {}).get("fen")
            if not fen:
                continue
            board = chess.Board(fen)
            infos = engine.analyse(
                board,
                chess.engine.Limit(depth=args.depth),
                multipv=args.multipv,
            )
            lines = []
            if isinstance(infos, dict):
                infos = [infos]
            for info in infos:
                pv = info.get("pv", [])
                score = info.get("score")
                san_line = pv_to_san(board, pv)
                lines.append(
                    {
                        "move": san_line[0] if san_line else None,
                        "pv": san_line,
                        "score": str(score) if score is not None else None,
                    }
                )
            row["stockfish"] = {"depth": args.depth, "multipv": args.multipv, "lines": lines}
            row["verification"] = {"status": "tool-labeled", "method": "stockfish"}
            label_lines = []
            for i, line in enumerate(lines, start=1):
                pv_text = " ".join(line["pv"]) if line["pv"] else "(none)"
                label_lines.append(f"{i}. {line['move']} | score {line['score']} | PV: {pv_text}")
            append_assistant_section(
                row,
                "Stockfish verification:\n"
                f"Depth {args.depth}, MultiPV {args.multipv}.\n" + "\n".join(label_lines),
            )
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

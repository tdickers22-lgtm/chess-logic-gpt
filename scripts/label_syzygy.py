#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import chess
import chess.syzygy

from chess_logic_gpt.records import read_jsonl


def wdl_description(wdl: int) -> str:
    return {
        2: "win for the side to move",
        1: "cursed win for the side to move",
        0: "draw with best play",
        -1: "blessed loss for the side to move",
        -2: "loss for the side to move",
    }.get(wdl, f"unknown WDL value {wdl}")


def append_assistant_section(row: dict, section: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            msg["content"] = str(msg.get("content", "")).rstrip() + "\n\n" + section
            return


def set_assistant_content(row: dict, content: str) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        return
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            msg["content"] = content
            return


def main() -> None:
    ap = argparse.ArgumentParser(description="Add local Syzygy WDL/DTZ labels to <=7 piece positions.")
    ap.add_argument("--infile", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--syzygy", default=os.environ.get("SYZYGY_PATH"))
    args = ap.parse_args()

    if not args.syzygy:
        raise SystemExit("Set --syzygy or SYZYGY_PATH.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with chess.syzygy.open_tablebase(args.syzygy) as tablebase, out.open("w", encoding="utf-8") as f:
        for row in read_jsonl(args.infile):
            fen = row.get("metadata", {}).get("fen")
            if not fen:
                continue
            board = chess.Board(fen)
            if len(board.piece_map()) > 7 or board.castling_rights:
                continue
            try:
                wdl = tablebase.probe_wdl(board)
                dtz = tablebase.probe_dtz(board)
                row["syzygy"] = {
                    "wdl": wdl,
                    "dtz": dtz,
                }
                row["verification"] = {"status": "verified", "method": "syzygy"}
                section = (
                    "Syzygy tablebase verification:\n"
                    f"WDL: {wdl} ({wdl_description(wdl)}).\n"
                    f"DTZ: {dtz}.\n"
                    "Interpretation: the label is exact for the given FEN from the side-to-move perspective. "
                    "Use WDL as the game-theoretic result and DTZ as the distance-to-zeroing guide under best play."
                )
                if row.get("task") == "syzygy_endgame_candidate":
                    set_assistant_content(row, section)
                else:
                    append_assistant_section(row, section)
            except chess.syzygy.MissingTableError:
                continue
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

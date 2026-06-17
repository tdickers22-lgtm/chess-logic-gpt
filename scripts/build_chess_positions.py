#!/usr/bin/env python3
from __future__ import annotations

import argparse

from tqdm import tqdm

from chess_logic_gpt.chess.examples import pgn_sample_to_record, sample_positions_from_game
from chess_logic_gpt.chess.pgn_stream import game_quality_ok, iter_games
from chess_logic_gpt.records import append_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream PGN(.zst) and build chess reasoning records.")
    ap.add_argument("--pgn", required=True, help="Path to .pgn or .pgn.zst")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit-games", type=int, default=10000)
    ap.add_argument("--min-elo", type=int, default=1800)
    ap.add_argument("--every-n-plies", type=int, default=8)
    ap.add_argument("--max-positions-per-game", type=int, default=8)
    args = ap.parse_args()

    written = 0
    for i, game in enumerate(tqdm(iter_games(args.pgn, args.limit_games), total=args.limit_games)):
        if not game_quality_ok(game, args.min_elo):
            continue
        samples = sample_positions_from_game(
            game,
            game_index=i,
            every_n_plies=args.every_n_plies,
            max_positions=args.max_positions_per_game,
        )
        for sample in samples:
            append_jsonl(args.out, pgn_sample_to_record(sample))
            written += 1
    print(f"wrote {written} records to {args.out}")


if __name__ == "__main__":
    main()


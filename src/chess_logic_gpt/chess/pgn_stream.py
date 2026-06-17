from __future__ import annotations

import io
from pathlib import Path
from typing import Iterator

import chess.pgn
import zstandard as zstd


def open_text_stream(path: str | Path):
    p = Path(path)
    raw = p.open("rb")
    if p.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(raw), encoding="utf-8", errors="replace")
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def iter_games(path: str | Path, limit: int | None = None) -> Iterator[chess.pgn.Game]:
    with open_text_stream(path) as stream:
        n = 0
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            yield game
            n += 1
            if limit is not None and n >= limit:
                break


def game_quality_ok(game: chess.pgn.Game, min_elo: int = 1800) -> bool:
    white_elo = _safe_int(game.headers.get("WhiteElo"))
    black_elo = _safe_int(game.headers.get("BlackElo"))
    if white_elo is None or black_elo is None:
        return False
    if min(white_elo, black_elo) < min_elo:
        return False
    result = game.headers.get("Result", "")
    return result in {"1-0", "0-1", "1/2-1/2"}


def _safe_int(value: str | None) -> int | None:
    try:
        if value is None or value == "?":
            return None
        return int(value)
    except ValueError:
        return None


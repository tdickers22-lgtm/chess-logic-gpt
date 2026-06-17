#!/usr/bin/env bash
set -euo pipefail

mkdir -p data/raw

cat <<'MSG'
This script intentionally downloads only a small historical sample by default.
For serious runs, edit LICHESS_URL to a recent month from https://database.lichess.org/.
MSG

LICHESS_URL="${LICHESS_URL:-https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst}"
OUT="data/raw/$(basename "$LICHESS_URL")"

if [ -f "$OUT" ]; then
  echo "Already exists: $OUT"
else
  curl -L "$LICHESS_URL" -o "$OUT"
fi

echo "$OUT"


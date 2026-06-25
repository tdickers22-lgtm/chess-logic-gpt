#!/usr/bin/env bash
set -u

PROJECT_DIR="/Users/tobiasdicker/ai-dev-system/projects/chess-logic-gpt"
STATE_DIR="/Users/tobiasdicker/.cache/chess-logic-gpt/kaggle-autolaunch"
LOG_FILE="$STATE_DIR/autolaunch.log"
MARKER="$STATE_DIR/started"

mkdir -p "$STATE_DIR"
exec >>"$LOG_FILE" 2>&1

echo "==== $(date -Iseconds) kaggle SFT autolaunch ===="

if [[ -f "$MARKER" ]]; then
  echo "already started: $(cat "$MARKER")"
  exit 0
fi

export PATH="/opt/anaconda3/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
cd "$PROJECT_DIR" || exit 1

status="$(kaggle kernels status tobiasdicker/chess-logic-gpt-sft-resume 2>&1 || true)"
echo "$status"
if echo "$status" | grep -Eiq 'RUNNING|PREPARING|PENDING|QUEUED|BUILDING'; then
  echo "$(date -Iseconds) existing active Kaggle run detected" > "$MARKER"
  exit 0
fi

out="$(kaggle kernels push -p kaggle_sft_resume 2>&1)"
rc=$?
echo "$out"

if [[ $rc -eq 0 ]]; then
  echo "$(date -Iseconds) kernel push accepted" > "$MARKER"
  exit 0
fi

if echo "$out" | grep -Fq "Maximum weekly GPU quota"; then
  echo "quota still exhausted; will retry"
  exit 0
fi

echo "unexpected kaggle push failure; will retry"
exit 0

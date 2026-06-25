#!/usr/bin/env bash
# Run the 8B GRPO phase on a raw GCP GPU VM.
#
# Required env:
#   CLG_RUN_BUCKET   gs:// bucket/prefix used for data in and checkpoints out
#
# Optional env:
#   CLG_RUN_ID       run folder name under $CLG_RUN_BUCKET/runs/
#   GRPO_CONFIG      training config path
#   CLG_SFT_ADAPTER  local SFT adapter checkpoint to continue
#   GRPO_MIX_SIZE    total rows for the verifiable GRPO mix
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CLG_RUN_BUCKET:?set CLG_RUN_BUCKET, e.g. gs://project-id-clg-runs}"
CLG_RUN_ID="${CLG_RUN_ID:-gcp-grpo-$(date -u +%Y%m%d-%H%M%S)}"
GRPO_CONFIG="${GRPO_CONFIG:-configs/training/qwen3_8b_grpo_gcp_45_30_25_sft1600.yaml}"
OUT_DIR="data/outputs/qwen3-8b-grpo-gcp-45-30-25-from-sft1600"
REMOTE_RUN="${CLG_RUN_BUCKET%/}/runs/${CLG_RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SFT_ADAPTER_DEFAULT="data/outputs/qwen3-8b-chess-logic-lora-45-30-25/checkpoint-1600"
export CLG_SFT_ADAPTER="${CLG_SFT_ADAPTER:-$SFT_ADAPTER_DEFAULT}"
GRPO_MIX_SIZE="${GRPO_MIX_SIZE:-24000}"
GRPO_SOURCE_DATA="${GRPO_SOURCE_DATA:-data/processed/train_mix_45_30_25.jsonl}"
GRPO_TRAIN_DATA="${GRPO_TRAIN_DATA:-data/processed/grpo_mix_45_30_25.jsonl}"
export GRPO_MIX_SIZE GRPO_SOURCE_DATA GRPO_TRAIN_DATA

mkdir -p data/processed "$OUT_DIR"

{
  echo "run_id=$CLG_RUN_ID"
  echo "config=$GRPO_CONFIG"
  echo "sft_adapter=$CLG_SFT_ADAPTER"
  echo "source_data=$GRPO_SOURCE_DATA"
  echo "train_data=$GRPO_TRAIN_DATA"
  echo "remote_run=$REMOTE_RUN"
  date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT_DIR/run.env"

if [[ "${CLG_SKIP_INSTALL:-0}" != "1" ]]; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install -e ".[training]"
  sudo "$PYTHON_BIN" -m pip uninstall -y torchaudio torchvision torchtext || true
fi

if [[ ! -s "$GRPO_SOURCE_DATA" ]]; then
  gcloud storage cp "${CLG_RUN_BUCKET%/}/data/train_mix_45_30_25.jsonl" "$GRPO_SOURCE_DATA"
fi

if [[ ! -d "$CLG_SFT_ADAPTER" ]]; then
  echo "missing SFT adapter: $CLG_SFT_ADAPTER" >&2
  exit 2
fi

if [[ ! -s "$GRPO_TRAIN_DATA" || "${GRPO_REBUILD_DATA:-0}" == "1" ]]; then
  "$PYTHON_BIN" - <<'PY'
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from chess_logic_gpt.eval import is_verifiable
from chess_logic_gpt.records import read_jsonl

source = Path(os.environ["GRPO_SOURCE_DATA"])
dest = Path(os.environ["GRPO_TRAIN_DATA"])
total = int(os.environ.get("GRPO_MIX_SIZE", "24000"))
targets = {
    "chess": round(total * 0.45),
    "logic": round(total * 0.30),
}
targets["memory"] = total - targets["chess"] - targets["logic"]

groups: dict[str, list[dict]] = defaultdict(list)
for record in read_jsonl(source):
    domain = record.get("domain", "")
    if domain in targets and is_verifiable(record):
        groups[domain].append(record)

rng = random.Random(1729)
selected: list[dict] = []
for domain, want in targets.items():
    rows = groups[domain]
    if len(rows) < want:
        raise SystemExit(f"not enough {domain} rows for GRPO mix: need {want}, have {len(rows)}")
    rng.shuffle(rows)
    selected.extend(rows[:want])

rng.shuffle(selected)
dest.parent.mkdir(parents=True, exist_ok=True)
with dest.open("w", encoding="utf-8") as f:
    for record in selected:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(
    "wrote",
    dest,
    len(selected),
    {domain: targets[domain] for domain in ("chess", "logic", "memory")},
    flush=True,
)
PY
fi

sync_once() {
  gcloud storage rsync -r "$OUT_DIR" "$REMOTE_RUN/$OUT_DIR" || true
}

sync_loop() {
  while true; do
    sync_once
    sleep 300
  done
}

sync_loop >"$OUT_DIR/gcs-sync.log" 2>&1 &
SYNC_PID=$!
trap 'kill "$SYNC_PID" 2>/dev/null || true; sync_once' EXIT

export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_XET_HIGH_PERFORMANCE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NO_TORCHAUDIO=1
export TOKENIZERS_PARALLELISM=false
export TRACKIO_PROJECT="${TRACKIO_PROJECT:-chess-logic-gpt}"
export TRACKIO_RUN="${TRACKIO_RUN:-grpo-gcp-45-30-25-sft1600}"

"$PYTHON_BIN" scripts/train_grpo.py --config "$GRPO_CONFIG" 2>&1 | tee "$OUT_DIR/train.log"
sync_once

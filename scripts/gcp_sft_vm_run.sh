#!/usr/bin/env bash
# Run the 8B SFT job on a raw GCP GPU VM.
#
# Required env:
#   CLG_RUN_BUCKET   gs:// bucket/prefix used for data in and checkpoints out
#
# Optional env:
#   CLG_RUN_ID       run folder name under $CLG_RUN_BUCKET/runs/
#   CLG_GIT_REF      git ref already checked out by the launcher
#   SFT_CONFIG       training config path
set -euo pipefail

cd "$(dirname "$0")/.."

: "${CLG_RUN_BUCKET:?set CLG_RUN_BUCKET, e.g. gs://project-id-clg-runs}"
CLG_RUN_ID="${CLG_RUN_ID:-gcp-sft-$(date -u +%Y%m%d-%H%M%S)}"
SFT_CONFIG="${SFT_CONFIG:-configs/training/qwen3_8b_lora_gcp_45_30_25.yaml}"
OUT_DIR="data/outputs/qwen3-8b-chess-logic-lora-45-30-25"
REMOTE_RUN="${CLG_RUN_BUCKET%/}/runs/${CLG_RUN_ID}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

mkdir -p data/processed "$OUT_DIR"

{
  echo "run_id=$CLG_RUN_ID"
  echo "git_ref=${CLG_GIT_REF:-unknown}"
  echo "config=$SFT_CONFIG"
  echo "remote_run=$REMOTE_RUN"
  date -u +"started_at=%Y-%m-%dT%H:%M:%SZ"
} | tee "$OUT_DIR/run.env"

"$PYTHON_BIN" -m pip install --upgrade pip
"$PYTHON_BIN" -m pip install -e ".[training]"
# The Deep Learning VM can ship mismatched optional audio/vision wheels. This
# text-only training path does not need them, and Transformers may import them
# opportunistically during module discovery.
sudo "$PYTHON_BIN" -m pip uninstall -y torchaudio torchvision torchtext || true

gcloud storage cp "${CLG_RUN_BUCKET%/}/data/train_mix_45_30_25.jsonl" \
  data/processed/train_mix_45_30_25.jsonl
gcloud storage cp "${CLG_RUN_BUCKET%/}/data/eval_mix_45_30_25.jsonl" \
  data/processed/eval_mix_45_30_25.jsonl

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
export TRACKIO_RUN="${TRACKIO_RUN:-sft-gcp-45-30-25}"

"$PYTHON_BIN" scripts/train_lora.py --config "$SFT_CONFIG" 2>&1 | tee "$OUT_DIR/train.log"
sync_once

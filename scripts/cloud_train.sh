#!/usr/bin/env bash
# Cloud-agnostic training entrypoint. Drives the SFT -> GRPO -> eval pipeline via
# env vars so the same command works on Azure ML, a raw GPU VM, RunPod, etc.
#
# Env vars:
#   PHASE        sft | grpo | eval | all   (default: sft)
#   SFT_CONFIG   default: configs/training/qwen_lora.yaml
#   GRPO_CONFIG  default: configs/training/qwen_grpo.yaml
#   HF_TOKEN     required for gated base models + push_to_hub + Trackio Space sync
#   TRACKIO_SPACE_ID   optional: mirror metrics to a Hugging Face Space dashboard
#   EVAL_MODEL / EVAL_ADAPTER / EVAL_DATA / EVAL_OUT   for PHASE=eval
#
# Big-model LoRA shards across all visible GPUs in ONE process (device_map=auto),
# so we launch with plain `python`, not multi-process DDP.
set -euo pipefail

cd "$(dirname "$0")/.."

export TRACKIO_PROJECT="${TRACKIO_PROJECT:-chess-logic-gpt}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

# Auto-stage data from the HF dataset repo if it's not already on disk
# (cloud-agnostic: a fresh GPU box only needs HF_TOKEN + CLG_HF_DATASET).
if [ -n "${CLG_HF_DATASET:-}" ] && [ ! -f "data/processed/train_mix.jsonl" ]; then
  echo ">>> staging data from HF dataset $CLG_HF_DATASET"
  python scripts/fetch_data.py --repo "$CLG_HF_DATASET"
fi

PHASE="${PHASE:-sft}"
SFT_CONFIG="${SFT_CONFIG:-configs/training/qwen_lora.yaml}"
GRPO_CONFIG="${GRPO_CONFIG:-configs/training/qwen_grpo.yaml}"

run_sft() {
  echo ">>> SFT phase (config=$SFT_CONFIG)"
  TRACKIO_RUN="${TRACKIO_RUN:-sft}" python scripts/train_lora.py --config "$SFT_CONFIG"
}

run_grpo() {
  echo ">>> GRPO phase (config=$GRPO_CONFIG)"
  TRACKIO_RUN="${TRACKIO_RUN:-grpo}" python scripts/train_grpo.py --config "$GRPO_CONFIG"
}

run_eval() {
  : "${EVAL_MODEL:?set EVAL_MODEL for PHASE=eval}"
  echo ">>> eval phase (model=$EVAL_MODEL)"
  python scripts/evaluate.py \
    --model "$EVAL_MODEL" \
    ${EVAL_ADAPTER:+--adapter "$EVAL_ADAPTER"} \
    --data "${EVAL_DATA:-data/processed/eval_mix.jsonl}" \
    --out "${EVAL_OUT:-data/outputs/eval_report.json}"
}

case "$PHASE" in
  sft)  run_sft ;;
  grpo) run_grpo ;;
  eval) run_eval ;;
  all)  run_sft; run_grpo ;;
  *)    echo "unknown PHASE='$PHASE' (expected sft|grpo|eval|all)" >&2; exit 2 ;;
esac

echo ">>> done (PHASE=$PHASE)"

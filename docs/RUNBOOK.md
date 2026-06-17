# Runbook

End-to-end pipeline: generate/ingest verifiable data → SFT (teach the
`<reasoning>/<answer>` format) → GRPO (RLVR, earn skill against verifiers) →
eval by motif/rating/recall → monitor with guardrails + alerts.

## 0. Secrets / API keys (the only things you must provide)

| Secret | Used for | How |
|--------|----------|-----|
| **Hugging Face token** (`HF_TOKEN`) | pull the base model, push the trained adapter, sync the Trackio dashboard | `modal secret create huggingface-secret HF_TOKEN=hf_xxx`, or pass as an env var on Azure/any box |
| **Compute account** (pick one) | remote GPU training | **Modal**: `modal token new`. **Azure** (GitHub Education credits): `az login` + an AmlCompute cluster (§6b). Or any rented GPU VM via the Docker image. |
| **Trackio Space id** (optional) | live training dashboard that persists after the box dies, e.g. `your-username/chess-logic-gpt` | pass `--trackio-space` to the Modal entrypoints |
| **Cursor API key** (`CURSOR_API_KEY`, optional) | let the monitor agent escalate failures to a Cursor background agent | export in the env where `scripts/monitor_training.py --escalate` runs |

Everything else runs with no credentials (Lichess puzzles are CC0, synthetic data is generated locally).

## 1. Local environment

```bash
cd /Users/tobiasdicker/ai-dev-system/projects/chess-logic-gpt
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,training,cloud]"
pytest -q && ruff check .
```

## 2. Generate clean synthetic data (logic + memory + ethics)

All generators now emit the shared trace skeleton
(`<reasoning> ... </reasoning>\n<answer> ... </answer>`).

```bash
python scripts/generate_logic.py --out data/processed/logic_synth.jsonl --n 50000 --seed 1
python scripts/generate_applied_reasoning.py --out data/processed/applied_reasoning.jsonl --n 50000 --seed 2
python scripts/generate_ethics_calibration.py --out data/processed/ethics_calibration.jsonl --n 50000 --seed 3
python scripts/generate_memory.py --out data/processed/memory_synth.jsonl --n 50000 --seed 4
```

(For a smoke test use `--n 200`.)

## 3. Chess data

### 3a. Lichess puzzles (primary tactics source, CC0) — motif drilling

Download + ingest a motif-balanced, repetition-heavy drilling set (Woodpecker
Method: same motifs across many distinct positions, spaced across passes):

```bash
python scripts/ingest_puzzles.py --infile data/raw/lichess_db_puzzle.csv.zst \
  --out data/processed/lichess_puzzles.jsonl \
  --min-rating 800 --max-rating 2200 \
  --curriculum --per-motif 3000 --repeat 3 --order blocked_then_interleaved
```

`--download` fetches `lichess_db_puzzle.csv.zst` (~290 MB) to `data/raw/` if you
don't have it. With `--curriculum` the script does a **memory-bounded full-file
harvest**: it streams the whole DB and keeps up to `per_motif * cap_factor`
distinct positions per core motif (validating with python-chess only until each
bucket fills), so even rare motifs are mined deeply without OOMing. Omit
`--limit` to scan the whole file. The current settings yield ~4.5k distinct
positions for each of the 7 common core motifs and ~2.3k for `backRankMate`,
i.e. a 72k-step drilling curriculum (8 motifs × 3000 × 3 spaced passes) over
~23k distinct positions. The script prints distinct-position counts per motif.

### 3b. (Optional) PGN positions, Stockfish labels, Syzygy endgames

```bash
bash scripts/download_lichess_sample.sh
python scripts/build_chess_positions.py --pgn data/raw/*.pgn.zst --out data/processed/chess_positions.jsonl --max-games 100000
STOCKFISH_PATH=/path/to/stockfish python scripts/label_stockfish.py --infile data/processed/chess_positions.jsonl --out data/processed/chess_positions.jsonl --depth 14 --multipv 3
python scripts/generate_endgame_candidates.py --out data/processed/endgame_candidates.jsonl --n 100000 --seed 4
SYZYGY_PATH=/path/to/syzygy python scripts/label_syzygy.py --infile data/processed/endgame_candidates.jsonl --out data/processed/endgames_syzygy.jsonl
```

### 3c. Held-out OOD puzzle eval (generalization, not memorization)

Build a *harder* eval band whose FENs are deduped against the training set, so
`by_motif`/`by_rating` accuracy measures whether tactical recognition generalises
rather than recalling drilled positions:

```bash
python scripts/build_ood_eval.py \
  --infile data/raw/lichess_db_puzzle.csv.zst \
  --train data/processed/lichess_puzzles.jsonl \
  --out data/processed/eval_puzzles_ood.jsonl \
  --min-rating 2300 --max-rating 2900 --per-motif 300
```

## 4. Mix the dataset

```bash
python scripts/mix_dataset.py --config configs/data/mix_50_25_25.yaml
# -> data/processed/train_mix.jsonl, data/processed/eval_mix.jsonl
```

### 4b. (Recommended) Push data to the Hub for cloud-agnostic staging

Host the prepared JSONL on a private HF dataset repo so any GPU box pulls it with
just `HF_TOKEN` (no per-platform upload dance):

```bash
HF_TOKEN=hf_xxx python scripts/push_data_to_hub.py --repo <user>/chess-logic-gpt-data
# on the GPU box (or automatically via cloud_train.sh when CLG_HF_DATASET is set):
HF_TOKEN=hf_xxx python scripts/fetch_data.py --repo <user>/chess-logic-gpt-data
```

`scripts/cloud_train.sh` auto-runs the fetch when `CLG_HF_DATASET=<user>/chess-logic-gpt-data`
is set and `data/processed/train_mix.jsonl` is missing.

## 5. SFT (Modal) — teach the format and base competence

Authenticate and upload data:

```bash
modal token new
modal secret create huggingface-secret HF_TOKEN=hf_xxx
modal volume create chess-logic-gpt-data
modal volume put chess-logic-gpt-data data/processed/train_mix.jsonl /processed/train_mix.jsonl
modal volume put chess-logic-gpt-data data/processed/eval_mix.jsonl /processed/eval_mix.jsonl
```

Cheap smoke first (0.5B), then the real run (`Qwen/Qwen3-30B-A3B`):

```bash
modal run cloud/modal/train_modal.py::sft --config /app/configs/training/smoke.yaml
modal run cloud/modal/train_modal.py::sft --config /app/configs/training/qwen_lora.yaml --trackio-space your-username/chess-logic-gpt
```

## 6. GRPO / RLVR (Modal) — earn skill against the verifiers

Point `model.base_model` in `configs/training/qwen_grpo.yaml` at the SFT
checkpoint, then:

```bash
modal run cloud/modal/train_modal.py::grpo --config /app/configs/training/qwen_grpo.yaml --trackio-space your-username/chess-logic-gpt
```

The reward is `chess_logic_gpt.rewards.grpo_reward` (legal-move / solution-line /
memory verifiers). Only verifiable records are used; applied/ethics rows are
skipped automatically.

## 6b. Alternative compute: Azure ML / any GPU box (no Modal)

The training stack is cloud-agnostic. `scripts/cloud_train.sh` drives the whole
pipeline from env vars (`PHASE=sft|grpo|eval|all`), and the bundled `Dockerfile`
(based on the official PyTorch CUDA image, `pip install -e .[training]`) runs
anywhere with NVIDIA GPUs.

**GitHub Education → Azure ML** (your likely path):

```bash
az extension add -n ml
az ml compute create -n gpu-a100 --type AmlCompute \
  --size Standard_NC24ads_A100_v4 --min-instances 0 --max-instances 1
# upload data/processed to the workspace datastore (path referenced in job.yml)
az ml job create -f cloud/azure/job.yml \
  --set environment_variables.HF_TOKEN=$HF_TOKEN \
  --set environment_variables.TRACKIO_SPACE_ID=your-username/chess-logic-gpt --web
```

**Any raw GPU VM / RunPod / Lambda** (no orchestration):

```bash
docker build -t chess-logic-gpt .
docker run --gpus all --rm \
  -e HF_TOKEN=hf_xxx -e PHASE=all \
  -e TRACKIO_SPACE_ID=your-username/chess-logic-gpt \
  -v "$PWD/data:/workspace/data" -v "$PWD/outputs:/workspace/data/outputs" \
  chess-logic-gpt
# or, on a box where you've already `pip install -e .[training]`:
PHASE=all HF_TOKEN=hf_xxx bash scripts/cloud_train.sh
```

Metrics (`metrics.jsonl` + Trackio) and the auto-stop `GuardrailCallback` work
identically regardless of where the job runs.

## 7. Evaluate (earned accuracy, not loss)

```bash
modal run cloud/modal/train_modal.py::evaluate --model /app/data/outputs/qwen-chess-logic-grpo
# or locally against a downloaded checkpoint:
python scripts/evaluate.py --model data/outputs/qwen-chess-logic-grpo --data data/processed/eval_mix.jsonl --out data/outputs/eval_report.json
# generalization check on the held-out, harder OOD band (FENs disjoint from train):
python scripts/evaluate.py --model data/outputs/qwen-chess-logic-grpo --data data/processed/eval_puzzles_ood.jsonl --out data/outputs/eval_report_ood.json
```

Report includes `by_motif`, `by_rating` (does tactical recognition generalise to
unseen, harder positions?) and `memory_recall_by_facts` (working-memory capacity
curve — now spanning 6→40-fact tables plus the graded `working_memory_multi_query`
"database" lookups).

## 8. Monitoring agents + alerts

Every run writes `<output_dir>/metrics.jsonl` on the volume and mirrors to
Trackio when `TRACKIO_SPACE_ID` is set. The in-process `GuardrailCallback` fires
alerts and **auto-stops on divergence** (NaN loss, grad explosion, etc.).

Run the external monitor agent (kill switch via a `STOP` sentinel the trainer
honours; optional Cursor-agent escalation):

```bash
# pull metrics from the volume, then watch them
modal volume get chess-logic-gpt-data /outputs/qwen-chess-logic-grpo/metrics.jsonl data/outputs/qwen-chess-logic-grpo/metrics.jsonl
python scripts/monitor_training.py \
  --metrics-file data/outputs/qwen-chess-logic-grpo/metrics.jsonl \
  --interval 30 --escalate
```

Guardrails: NaN/Inf loss, loss spike, loss stall, reward collapse, reward floor,
KL blowup, grad explosion, eval regression, and reward/eval divergence (possible
reward hacking).

## 9. Verify the base model id

```bash
hf model info Qwen/Qwen3-30B-A3B   # 30.5B total / 3.3B active MoE, Apache-2.0
```

Swap to `Qwen/Qwen3-30B-A3B-Instruct-2507` if you prefer the updated instruct post-train.

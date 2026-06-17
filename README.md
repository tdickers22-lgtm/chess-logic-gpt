# Chess Logic GPT

Clean-room pipeline for building a chess + logic reasoning model.

Goal:
- 50% chess deduction, PGN continuation, tactical motifs, endgame reasoning.
- 25% formal logic, Fitch-style natural deduction, predicate logic, proof repair, applied reasoning across philosophical/social domains, and ethics calibration that reduces over-refusal.
- 25% working-memory, constraint puzzles, IQ-style synthetic tasks, recall drills.

The model should learn pattern recognition and deductive habits in its weights while
exact knowledge stays in retrieval/tools:
- In weights: motifs, proof habits, endgame principles, calculation style.
- In retrieval/tools: exact PGNs, exact book notes, tablebase facts, engine lines.

## Clean Data Rule

Use only data that is:
- public-domain,
- permissively licensed,
- explicitly allowed for this kind of processing,
- self-generated and verifier-checked,
- or privately owned/licensed by you.

Do not ingest copyrighted chess books, proprietary puzzle books, commercial courses,
or scraped private content. For books such as Silman, use your own notes or licensed
material, not raw book text.

## Pipeline

```text
raw sources (Lichess puzzles CC0, synthetic logic/memory, optional PGN/Stockfish/Syzygy)
  -> source manifest + license metadata
  -> example builders emit a shared <reasoning>/<answer> trace
  -> verifier checks (python-chess legal moves, solution lines, memory, Fitch proofs)
  -> motif-weighted curriculum (massed + spaced repetition)
  -> mixed JSONL train/eval sets
  -> SFT (teach the format) -> GRPO/RLVR (earn skill against the verifiers)
  -> eval by motif/rating/recall  +  guardrail monitor (alerts, auto-stop, agent escalation)
```

## Recommended Base Model

Start with a Qwen open-weight MoE reasoning model. The default config targets
`Qwen/Qwen3-30B-A3B` (30.5B total / 3.3B active, Apache-2.0) as the first serious
run; `configs/training/smoke.yaml` uses `Qwen/Qwen2.5-0.5B-Instruct` for cheap
end-to-end smoke tests.

## Quick Start

```bash
cd ai-dev-system/projects/chess-logic-gpt
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,training]"

# Generate small clean synthetic datasets locally.
python scripts/generate_logic.py --out data/processed/logic_synth.jsonl --n 1000
python scripts/generate_applied_reasoning.py --out data/processed/applied_reasoning.jsonl --n 1000
python scripts/generate_ethics_calibration.py --out data/processed/ethics_calibration.jsonl --n 1000
python scripts/generate_memory.py --out data/processed/memory_synth.jsonl --n 1000

# Optional: create exact endgame truth data if you have Syzygy files locally.
python scripts/generate_endgame_candidates.py --out data/processed/endgame_candidates.jsonl --n 1000
SYZYGY_PATH=/path/to/syzygy \
  python scripts/label_syzygy.py \
  --infile data/processed/endgame_candidates.jsonl \
  --out data/processed/endgames_syzygy.jsonl

# Mix whatever processed datasets exist.
python scripts/mix_dataset.py \
  --config configs/data/mix_50_25_25.yaml \
  --out data/processed/train_mix.jsonl

# Optional smoke test.
pytest
```

## Big Data Sources

Use `configs/data/sources.clean.yaml` as the authoritative source list. The initial
allowlist includes Lichess games, Lichess puzzles, Lichess broadcasts, Syzygy/Stockfish
labels, Open Logic Project material, LeanDojo/Lean theorem traces, and generated
synthetic tasks.

See `docs/DATASETS.md` for the full dataset plan and `docs/RUNBOOK.md` for commands.

## Remote Training

This repo includes:
- `src/chess_logic_gpt/training/train_lora.py`: generic Hugging Face/PEFT LoRA trainer.
- `cloud/modal/train_modal.py`: Modal entrypoint for remote GPU training.
- `configs/training/qwen_lora.yaml`: default LoRA config.

You still need a cloud account/API token. Do not commit secrets.

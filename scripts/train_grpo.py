#!/usr/bin/env python3
"""GRPO (RLVR) training wired to the verifiable reward.

This is where genuine *skill* is acquired: the model samples several candidate
solutions per prompt, each is scored by `rewards.grpo_reward` (legal-move /
solution-line / memory verifiers -- no human labels, no separate reward model),
and GRPO pushes probability toward the higher-reward samples.

Run SFT first (`scripts/train_lora.py`) so the model already emits the
<reasoning>/<answer> format, then point `model.base_model` here at that
checkpoint. Only verifiable records (chess tactics, formal-logic proofs, memory)
are used; applied/ethics SFT data is skipped automatically.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from chess_logic_gpt.records import read_jsonl
from chess_logic_gpt.rewards import grpo_reward
from chess_logic_gpt.training.callbacks import GuardrailCallback
from chess_logic_gpt.training.formatting import render_chat
from chess_logic_gpt.training.grpo_data import build_prompt_rows
from chess_logic_gpt.training.monitoring import MetricLogger


def build_prompt_dataset(path: str, tokenizer) -> Dataset:  # noqa: ANN001
    def render(messages: list[dict]) -> str:
        return render_chat(tokenizer, messages, add_generation_prompt=True)

    rows = build_prompt_rows(read_jsonl(path), render)
    if not rows:
        raise SystemExit(f"no verifiable records in {path}; ingest puzzles / generate logic+memory first")
    return Dataset.from_list(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="GRPO RLVR training on verifiable rewards.")
    ap.add_argument("--config", default="configs/training/qwen_grpo.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    grpo_cfg = cfg["grpo"]
    lora_cfg = cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = build_prompt_dataset(data_cfg["train_file"], tokenizer)

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )

    # vLLM generation is load-bearing for an 8B+ GRPO run (8 samples x ~1k tokens
    # per prompt every step); pass the keys only when present so older TRL still works.
    vllm_kwargs = {
        key: grpo_cfg[key]
        for key in ("use_vllm", "vllm_mode", "vllm_gpu_memory_utilization")
        if key in grpo_cfg
    }

    output_dir = grpo_cfg["output_dir"]
    config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=float(grpo_cfg.get("learning_rate", 1e-6)),
        per_device_train_batch_size=int(grpo_cfg.get("per_device_train_batch_size", 4)),
        gradient_accumulation_steps=int(grpo_cfg.get("gradient_accumulation_steps", 4)),
        num_generations=int(grpo_cfg.get("num_generations", 8)),
        max_prompt_length=int(grpo_cfg.get("max_prompt_length", 1024)),
        max_completion_length=int(grpo_cfg.get("max_completion_length", 1024)),
        num_train_epochs=float(grpo_cfg.get("num_train_epochs", 1)),
        max_steps=int(grpo_cfg.get("max_steps", -1)),
        beta=float(grpo_cfg.get("beta", 0.04)),
        temperature=float(grpo_cfg.get("temperature", 1.0)),
        logging_steps=int(grpo_cfg.get("logging_steps", 1)),
        save_steps=int(grpo_cfg.get("save_steps", 100)),
        save_total_limit=int(grpo_cfg.get("save_total_limit", 3)),
        bf16=bool(grpo_cfg.get("bf16", True)),
        gradient_checkpointing=bool(grpo_cfg.get("gradient_checkpointing", True)),
        report_to="none",  # GuardrailCallback owns Trackio logging
        push_to_hub=bool(model_cfg.get("push_to_hub", False)),
        hub_model_id=model_cfg.get("hub_model_id"),
        **vllm_kwargs,
    )

    logger = MetricLogger(
        Path(output_dir) / "metrics.jsonl",
        project=os.environ.get("TRACKIO_PROJECT", "chess-logic-gpt"),
        run=os.environ.get("TRACKIO_RUN", "grpo"),
        space_id=os.environ.get("TRACKIO_SPACE_ID"),
        config={"phase": "grpo", **{k: grpo_cfg.get(k) for k in ("learning_rate", "beta", "num_generations")}},
    )
    callback = GuardrailCallback(logger, stop_file=str(Path(output_dir) / "STOP"))

    trainer = GRPOTrainer(
        model=model_cfg["base_model"],
        args=config,
        train_dataset=dataset,
        reward_funcs=[grpo_reward],
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=[callback],
    )
    resume = grpo_cfg.get("resume_from_checkpoint", False)
    if resume is True:
        from transformers.trainer_utils import get_last_checkpoint

        resume = get_last_checkpoint(output_dir) if Path(output_dir).is_dir() else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    main()

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
import inspect
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
    desired = {
        "output_dir": output_dir,
        "learning_rate": float(grpo_cfg.get("learning_rate", 1e-6)),
        "per_device_train_batch_size": int(grpo_cfg.get("per_device_train_batch_size", 4)),
        "gradient_accumulation_steps": int(grpo_cfg.get("gradient_accumulation_steps", 4)),
        "num_generations": int(grpo_cfg.get("num_generations", 8)),
        "max_prompt_length": int(grpo_cfg.get("max_prompt_length", 1024)),
        "max_completion_length": int(grpo_cfg.get("max_completion_length", 1024)),
        "num_train_epochs": float(grpo_cfg.get("num_train_epochs", 1)),
        "max_steps": int(grpo_cfg.get("max_steps", -1)),
        "beta": float(grpo_cfg.get("beta", 0.04)),
        "temperature": float(grpo_cfg.get("temperature", 1.0)),
        "logging_steps": int(grpo_cfg.get("logging_steps", 1)),
        "save_steps": int(grpo_cfg.get("save_steps", 100)),
        "save_total_limit": int(grpo_cfg.get("save_total_limit", 3)),
        "bf16": bool(grpo_cfg.get("bf16", True)),
        "gradient_checkpointing": bool(grpo_cfg.get("gradient_checkpointing", True)),
        "report_to": "none",  # GuardrailCallback owns Trackio logging
        "push_to_hub": bool(model_cfg.get("push_to_hub", False)),
        "hub_model_id": model_cfg.get("hub_model_id"),
        **vllm_kwargs,
    }
    # GRPOConfig's accepted arguments vary across TRL versions (e.g. max_prompt_length
    # was renamed/removed); keep only what this installed version accepts.
    accepted = set(inspect.signature(GRPOConfig).parameters) | set(getattr(GRPOConfig, "__dataclass_fields__", {}))
    dropped = sorted(k for k in desired if k not in accepted)
    if dropped:
        print("GRPOConfig: dropping args unsupported by this TRL version:", dropped, flush=True)
    config = GRPOConfig(**{k: v for k, v in desired.items() if k in accepted})

    logger = MetricLogger(
        Path(output_dir) / "metrics.jsonl",
        project=os.environ.get("TRACKIO_PROJECT", "chess-logic-gpt"),
        run=os.environ.get("TRACKIO_RUN", "grpo"),
        space_id=os.environ.get("TRACKIO_SPACE_ID"),
        config={"phase": "grpo", **{k: grpo_cfg.get(k) for k in ("learning_rate", "beta", "num_generations")}},
    )
    callback = GuardrailCallback(logger, stop_file=str(Path(output_dir) / "STOP"))

    # Continue from the SFT skill. Two paths depending on the GPU:
    #  - load_in_4bit (fits a 24GB L4): keep the base 4-bit and CONTINUE training
    #    the SFT adapter directly (no merge) -- pass the peft model, no peft_config.
    #  - else (A100): merge the SFT LoRA into the base, GRPO a fresh LoRA on top.
    model_for_trainer = model_cfg["base_model"]
    peft_for_trainer = peft_config
    # CLG_SFT_ADAPTER lets a runner (e.g. a Kaggle kernel) point at a locally
    # attached adapter dir instead of a private HF repo, avoiding a token.
    sft_adapter = os.environ.get("CLG_SFT_ADAPTER") or model_cfg.get("sft_adapter")
    if sft_adapter:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        load_4bit = bool(model_cfg.get("load_in_4bit", False))
        quant = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            if load_4bit
            else None
        )
        base = AutoModelForCausalLM.from_pretrained(
            model_cfg["base_model"],
            torch_dtype=torch.bfloat16,
            quantization_config=quant,
            trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
            device_map={"": 0},
        )
        if load_4bit:
            base = prepare_model_for_kbit_training(base)
            model_for_trainer = PeftModel.from_pretrained(base, sft_adapter, is_trainable=True)
            peft_for_trainer = None  # train the existing adapter, don't add a fresh one
        else:
            model_for_trainer = PeftModel.from_pretrained(base, sft_adapter).merge_and_unload()

    trainer = GRPOTrainer(
        model=model_for_trainer,
        args=config,
        train_dataset=dataset,
        reward_funcs=[grpo_reward],
        peft_config=peft_for_trainer,
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

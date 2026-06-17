from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from chess_logic_gpt.training.callbacks import GuardrailCallback
from chess_logic_gpt.training.formatting import build_supervised_example
from chess_logic_gpt.training.monitoring import MetricLogger


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a LoRA adapter on mixed chess/logic/memory JSONL.")
    ap.add_argument("--config", default="configs/training/qwen_lora.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))

    model_cfg = cfg["model"]
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    lora_cfg = cfg["lora"]

    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = None
    if model_cfg.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model"],
        trust_remote_code=bool(model_cfg.get("trust_remote_code", True)),
        quantization_config=quantization_config,
        device_map="auto",
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg["dropout"]),
        target_modules=list(lora_cfg["target_modules"]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    files = {"train": data_cfg["train_file"]}
    eval_file = data_cfg.get("eval_file")
    if eval_file and Path(eval_file).exists():
        files["validation"] = eval_file
    ds = load_dataset("json", data_files=files)

    max_len = int(data_cfg.get("max_seq_length", 4096))

    def batch_to_rows(batch: dict[str, list]) -> list[dict]:
        keys = list(batch.keys())
        return [dict(zip(keys, values)) for values in zip(*(batch[key] for key in keys))]

    def tokenize(batch):
        examples = [
            build_supervised_example(row["messages"], tokenizer, max_len)
            for row in batch_to_rows(batch)
        ]
        return {
            "input_ids": [e["input_ids"] for e in examples],
            "attention_mask": [e["attention_mask"] for e in examples],
            "labels": [e["labels"] for e in examples],
        }

    tokenized = ds.map(
        tokenize,
        batched=True,
        remove_columns=ds["train"].column_names,
    )

    args_out = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1)),
        max_steps=int(train_cfg.get("max_steps", -1)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 8e-5)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.03)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_steps=int(train_cfg.get("eval_steps", 200)),
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=int(train_cfg.get("save_total_limit", 3)),
        bf16=bool(train_cfg.get("bf16", True)),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        report_to="none",  # GuardrailCallback owns Trackio logging
        eval_strategy="steps" if "validation" in tokenized else "no",
    )

    logger = MetricLogger(
        Path(train_cfg["output_dir"]) / "metrics.jsonl",
        project=os.environ.get("TRACKIO_PROJECT", "chess-logic-gpt"),
        run=os.environ.get("TRACKIO_RUN", "sft"),
        space_id=os.environ.get("TRACKIO_SPACE_ID"),
        config={"phase": "sft", "base_model": model_cfg["base_model"]},
    )
    trainer = Trainer(
        model=model,
        args=args_out,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer, model=model, label_pad_token_id=-100, padding=True
        ),
        callbacks=[GuardrailCallback(logger, stop_file=str(Path(train_cfg["output_dir"]) / "STOP"))],
    )
    # Resume from the last checkpoint when asked, so an interrupted free-tier
    # session (Kaggle/Lightning preemption, 12h cap) picks up where it left off.
    resume = train_cfg.get("resume_from_checkpoint", False)
    if resume is True:
        from transformers.trainer_utils import get_last_checkpoint

        out_dir = train_cfg["output_dir"]
        resume = get_last_checkpoint(out_dir) if Path(out_dir).is_dir() else None
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(train_cfg["output_dir"])
    tokenizer.save_pretrained(train_cfg["output_dir"])
    if model_cfg.get("hub_model_id"):
        trainer.push_to_hub(model_cfg["hub_model_id"])


if __name__ == "__main__":
    main()

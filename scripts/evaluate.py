#!/usr/bin/env python3
"""Evaluate a model on the verifiable eval set with earned-accuracy breakdowns.

Reports overall accuracy plus by-motif and by-rating tactical accuracy (does
pattern recognition generalise to unseen positions and harder puzzles?) and a
memory recall-vs-table-size curve. Numbers come from the same verifier used for
RL, so there's no train/eval reward mismatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chess_logic_gpt.eval import evaluate, is_verifiable
from chess_logic_gpt.records import read_jsonl
from chess_logic_gpt.training.formatting import render_chat
from chess_logic_gpt.training.precision import dtype_from_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a model on verifiable records.")
    ap.add_argument("--model", required=True, help="Base model dir or HF hub id")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter dir")
    ap.add_argument("--data", default="data/processed/eval_mix.jsonl")
    ap.add_argument("--out", default="data/outputs/eval_report.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--load-in-4bit", action="store_true", help="QLoRA-style 4-bit base (fits 16GB GPUs)")
    ap.add_argument("--shuffle", action="store_true", help="shuffle before --limit for a representative sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--debug", type=int, default=0, help="print raw model output + score for the first N records")
    ap.add_argument("--batch-size", type=int, default=1, help="number of prompts to generate per forward pass")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quant = None
    dtype = dtype_from_config("auto")
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig

        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype,
        quantization_config=quant, device_map="auto",
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def _generate_from_prompts(prompts: list[str]) -> list[str]:
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
        with torch.no_grad():
            out = model.generate(**inputs, **generation_kwargs)
        prompt_width = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(row[prompt_width:], skip_special_tokens=True)
            for row in out
        ]

    def generate(record: dict) -> str:
        prompt = render_chat(tokenizer, record["messages"][:-1], add_generation_prompt=True)
        return _generate_from_prompts([prompt])[0]

    def generate_batch(batch: list[dict]) -> list[str]:
        prompts = [
            render_chat(tokenizer, record["messages"][:-1], add_generation_prompt=True)
            for record in batch
        ]
        return _generate_from_prompts(prompts)

    records = list(read_jsonl(args.data))
    if args.shuffle:
        import random

        random.Random(args.seed).shuffle(records)
    if args.limit:
        records = records[: args.limit]

    if args.debug:
        from chess_logic_gpt.rewards import score as _score

        for rec in records[: args.debug]:
            out = generate(rec)
            res = _score(rec, out)
            md = rec.get("metadata", {})
            print("=" * 70, flush=True)
            print("motif:", md.get("primary_motif"), "| gold line_uci:", md.get("line_uci"), flush=True)
            print("RAW OUTPUT:", repr(out[:600]), flush=True)
            print("SCORE:", res.score, "|", res.detail, flush=True)

    if args.batch_size <= 1:
        report = evaluate(records, generate)
    else:
        verifiable = [record for record in records if is_verifiable(record)]
        outputs: dict[int, str] = {}
        total = len(verifiable)
        print(f"evaluating {total} verifiable records with batch_size={args.batch_size}", flush=True)
        for start in range(0, total, args.batch_size):
            batch = verifiable[start : start + args.batch_size]
            for record, output in zip(batch, generate_batch(batch), strict=True):
                outputs[id(record)] = output
            print(f"generated {min(start + len(batch), total)}/{total}", flush=True)

        # Reuse the canonical harness aggregation so batched and serial reports match.
        report = evaluate(records, lambda record: outputs[id(record)])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote report to {out_path}")


if __name__ == "__main__":
    main()

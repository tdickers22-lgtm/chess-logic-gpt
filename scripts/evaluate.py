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

from chess_logic_gpt.eval import evaluate
from chess_logic_gpt.records import read_jsonl
from chess_logic_gpt.training.formatting import render_chat


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate a model on verifiable records.")
    ap.add_argument("--model", required=True, help="Base model dir or HF hub id")
    ap.add_argument("--adapter", default=None, help="Optional LoRA adapter dir")
    ap.add_argument("--data", default="data/processed/eval_mix.jsonl")
    ap.add_argument("--out", default="data/outputs/eval_report.json")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    def generate(record: dict) -> str:
        prompt = render_chat(tokenizer, record["messages"][:-1], add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=max(args.temperature, 1e-5),
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    records = list(read_jsonl(args.data))
    if args.limit:
        records = records[: args.limit]
    report = evaluate(records, generate)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote report to {out_path}")


if __name__ == "__main__":
    main()

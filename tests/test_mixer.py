from __future__ import annotations

import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from chess_logic_gpt.records import write_jsonl


def make_rows(domain: str, n: int) -> list[dict]:
    return [
        {
            "id": f"{domain}-{i}",
            "domain": domain,
            "task": "test",
            "source": {"name": "test", "url": "generated", "license": "test", "provenance": "generated"},
            "messages": [
                {"role": "system", "content": "system"},
                {"role": "user", "content": f"{domain} {i}"},
                {"role": "assistant", "content": "answer"},
            ],
            "verification": {"status": "verified", "method": "test"},
        }
        for i in range(n)
    ]


def test_mix_dataset_script_respects_domain_weights(tmp_path: Path) -> None:
    chess_path = tmp_path / "chess.jsonl"
    logic_path = tmp_path / "logic.jsonl"
    memory_path = tmp_path / "memory.jsonl"
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    config_path = tmp_path / "mix.yaml"

    write_jsonl(chess_path, make_rows("chess", 4))
    write_jsonl(logic_path, make_rows("logic", 4))
    write_jsonl(memory_path, make_rows("memory", 4))
    config_path.write_text(
        f"""
version: 1
seed: 1
target_mix:
  chess: 0.50
  logic: 0.25
  memory: 0.25
inputs:
  chess:
    - {chess_path}
  logic:
    - {logic_path}
  memory:
    - {memory_path}
output:
  train: {train_path}
  eval: {eval_path}
eval_fraction: 0
max_records: 8
""",
        encoding="utf-8",
    )

    subprocess.run(
        [sys.executable, "scripts/mix_dataset.py", "--config", str(config_path)],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )

    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines()]
    counts = Counter(row["domain"] for row in rows)
    assert counts == {"chess": 4, "logic": 2, "memory": 2}

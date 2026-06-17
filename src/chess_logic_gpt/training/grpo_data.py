"""Torch-free construction of the GRPO prompt dataset.

Kept free of `trl`/`datasets`/`torch` so the RL *data contract* -- record ->
prompt row (carrying domain/task/meta) -> `grpo_reward` -> scalar -- can be unit
tested without a GPU stack. `scripts/train_grpo.py` wraps these rows in a
`datasets.Dataset`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable

from chess_logic_gpt.eval import is_verifiable

RenderPrompt = Callable[[list[dict]], str]


def build_prompt_rows(records: Iterable[dict], render_prompt: RenderPrompt) -> list[dict]:
    """Build GRPO prompt rows from records, keeping only verifiable ones.

    Each row carries the columns `grpo_reward` needs to reconstruct a record and
    score a sampled completion: ``prompt``, ``domain``, ``task``, ``meta`` (JSON).
    """
    rows: list[dict] = []
    for record in records:
        if not is_verifiable(record):
            continue
        messages = record["messages"]
        rows.append(
            {
                "prompt": render_prompt(messages[:-1]),
                "domain": record.get("domain", ""),
                "task": record.get("task", ""),
                "meta": json.dumps(record.get("metadata", {}), ensure_ascii=False),
            }
        )
    return rows

"""Shared reasoning-trace skeleton used by every generator and domain.

One format across chess, logic, and memory so the model learns a single,
domain-general procedure (work in <reasoning>, commit in <answer>) that
transfers between domains. The reward/eval layer parses <answer> with
`chess_logic_gpt.rewards.final_answer`, so this is the single contract that
ties generation, verification, and RL together.
"""

from __future__ import annotations

REASONING_OPEN, REASONING_CLOSE = "<reasoning>", "</reasoning>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"


def wrap_trace(reasoning: str, answer: str) -> str:
    return (
        f"{REASONING_OPEN}\n{reasoning.strip()}\n{REASONING_CLOSE}\n"
        f"{ANSWER_OPEN}\n{answer.strip()}\n{ANSWER_CLOSE}"
    )

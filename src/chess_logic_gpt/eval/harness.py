"""Out-of-distribution evaluation harness built on the verifiable rewards.

The same verifier that drives RL also measures the model, so eval numbers are
*earned move accuracy*, not loss. The harness is deliberately torch-free: it
takes a ``generate(record) -> str`` callable, so it can be unit-tested with a
stub and driven in production by a real HF model (see ``scripts/evaluate.py``).

Breakdowns answer the questions that matter for the north star:
- ``by_motif`` / ``by_rating``: did tactical *pattern recognition* generalise to
  unseen positions, motif by motif and harder and harder?
- ``memory_recall_by_facts``: how does recall hold up as the table grows (the
  working-memory capacity curve)?
"""

from __future__ import annotations

from collections import defaultdict
from statistics import mean

from chess_logic_gpt.rewards import RewardResult, final_answer, score


def gold_answer(record: dict) -> str:
    return final_answer(record["messages"][-1]["content"])


def is_verifiable(record: dict) -> bool:
    domain = record.get("domain")
    if domain in ("chess", "memory"):
        return True
    if domain == "logic" and "goal" in record.get("metadata", {}):
        return True
    return False


def _rating_bucket(rating: int, width: int = 200) -> str:
    base = (int(rating) // width) * width
    return f"{base}-{base + width - 1}"


def _agg(subset: list[tuple[dict, RewardResult]]) -> dict:
    if not subset:
        return {"n": 0, "accuracy": 0.0, "mean_score": 0.0}
    return {
        "n": len(subset),
        "accuracy": mean(1.0 if r.correct else 0.0 for _, r in subset),
        "mean_score": mean(r.score for _, r in subset),
    }


def evaluate(records: list[dict], generate) -> dict:
    """Score model generations over the verifiable records and aggregate.

    generate: callable taking a record and returning the model's raw output
              (expected to contain an <answer>...</answer> block).
    """
    detailed: list[tuple[dict, RewardResult]] = []
    for record in records:
        if not is_verifiable(record):
            continue
        detailed.append((record, score(record, generate(record))))

    by_domain: dict[str, list] = defaultdict(list)
    by_motif: dict[str, list] = defaultdict(list)
    by_rating: dict[str, list] = defaultdict(list)
    by_facts: dict[int, list] = defaultdict(list)

    for record, result in detailed:
        domain = record.get("domain")
        by_domain[domain].append((record, result))
        md = record.get("metadata", {})
        if domain == "chess":
            by_motif[md.get("primary_motif", "unknown")].append((record, result))
            by_rating[_rating_bucket(md.get("rating", 0))].append((record, result))
        elif domain == "memory":
            n_facts = md.get("n_facts")
            if n_facts is not None:
                by_facts[int(n_facts)].append((record, result))

    return {
        "overall": _agg(detailed),
        "by_domain": {k: _agg(v) for k, v in sorted(by_domain.items())},
        "by_motif": {k: _agg(v) for k, v in sorted(by_motif.items())},
        "by_rating": {k: _agg(v) for k, v in sorted(by_rating.items())},
        "memory_recall_by_facts": {str(k): _agg(v) for k, v in sorted(by_facts.items())},
    }

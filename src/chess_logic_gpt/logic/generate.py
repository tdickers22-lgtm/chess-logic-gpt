from __future__ import annotations

import random
from dataclasses import dataclass

from chess_logic_gpt.records import stable_id
from chess_logic_gpt.training.trace import wrap_trace


ATOM_POOL = ["P", "Q", "R", "S", "T", "A", "B", "C"]
PREDICATES = ["Student", "Logician", "ChessPlayer", "Careful", "Rational", "Mortal"]


@dataclass(frozen=True)
class LogicExample:
    task: str
    premises: list[str]
    goal: str
    proof_lines: list[tuple[str, str]]
    explanation: str


def generate_logic_examples(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    examples: list[dict] = []
    templates = [
        make_modus_ponens,
        make_hypothetical_syllogism,
        make_conjunction_elim,
        make_conjunction_intro,
        make_disjunctive_syllogism,
        make_universal_instantiation,
        make_universal_chain,
    ]
    for i in range(n):
        ex = rng.choice(templates)(rng)
        examples.append(to_record(ex, i))
    return examples


def make_modus_ponens(rng: random.Random) -> LogicExample:
    p, q = rng.sample(ATOM_POOL, 2)
    return LogicExample(
        task="fitch_modus_ponens",
        premises=[f"{p} -> {q}", p],
        goal=q,
        proof_lines=[
            (f"{p} -> {q}", "Premise"),
            (p, "Premise"),
            (q, "->E 1,2"),
        ],
        explanation=f"Since {p} implies {q}, and {p} is given, {q} follows by conditional elimination.",
    )


def make_hypothetical_syllogism(rng: random.Random) -> LogicExample:
    p, q, r = rng.sample(ATOM_POOL, 3)
    return LogicExample(
        task="fitch_hypothetical_syllogism",
        premises=[f"{p} -> {q}", f"{q} -> {r}", p],
        goal=r,
        proof_lines=[
            (f"{p} -> {q}", "Premise"),
            (f"{q} -> {r}", "Premise"),
            (p, "Premise"),
            (q, "->E 1,3"),
            (r, "->E 2,4"),
        ],
        explanation=f"Apply the first conditional to get {q}; then apply the second conditional to get {r}.",
    )


def make_conjunction_elim(rng: random.Random) -> LogicExample:
    p, q = rng.sample(ATOM_POOL, 2)
    goal = rng.choice([p, q])
    return LogicExample(
        task="fitch_conjunction_elimination",
        premises=[f"{p} & {q}"],
        goal=goal,
        proof_lines=[
            (f"{p} & {q}", "Premise"),
            (goal, "&E 1"),
        ],
        explanation=f"A conjunction entails each conjunct, so {goal} follows from {p} & {q}.",
    )


def make_conjunction_intro(rng: random.Random) -> LogicExample:
    p, q = rng.sample(ATOM_POOL, 2)
    return LogicExample(
        task="fitch_conjunction_introduction",
        premises=[p, q],
        goal=f"{p} & {q}",
        proof_lines=[
            (p, "Premise"),
            (q, "Premise"),
            (f"{p} & {q}", "&I 1,2"),
        ],
        explanation=f"Both {p} and {q} are available, so their conjunction follows.",
    )


def make_disjunctive_syllogism(rng: random.Random) -> LogicExample:
    p, q = rng.sample(ATOM_POOL, 2)
    return LogicExample(
        task="fitch_disjunctive_syllogism",
        premises=[f"{p} v {q}", f"~{p}"],
        goal=q,
        proof_lines=[
            (f"{p} v {q}", "Premise"),
            (f"~{p}", "Premise"),
            (q, "vE/DS 1,2"),
        ],
        explanation=f"The disjunction says {p} or {q}; {p} is ruled out, so {q} remains.",
    )


def make_universal_instantiation(rng: random.Random) -> LogicExample:
    pred = rng.choice(PREDICATES)
    name = rng.choice(["a", "b", "c", "socrates", "magnus"])
    return LogicExample(
        task="predicate_universal_elimination",
        premises=[f"forall x {pred}(x)"],
        goal=f"{pred}({name})",
        proof_lines=[
            (f"forall x {pred}(x)", "Premise"),
            (f"{pred}({name})", "forallE 1"),
        ],
        explanation=f"A universal statement applies to every object, including {name}.",
    )


def make_universal_chain(rng: random.Random) -> LogicExample:
    p, q = rng.sample(PREDICATES, 2)
    name = rng.choice(["a", "b", "c", "socrates", "magnus"])
    return LogicExample(
        task="predicate_universal_modus_ponens",
        premises=[f"forall x ({p}(x) -> {q}(x))", f"{p}({name})"],
        goal=f"{q}({name})",
        proof_lines=[
            (f"forall x ({p}(x) -> {q}(x))", "Premise"),
            (f"{p}({name})", "Premise"),
            (f"{p}({name}) -> {q}({name})", "forallE 1"),
            (f"{q}({name})", "->E 3,2"),
        ],
        explanation=f"Instantiate the universal conditional at {name}, then use modus ponens.",
    )


def to_record(ex: LogicExample, index: int) -> dict:
    prompt = (
        "Construct a Fitch-style proof. Use explicit line numbers and rule citations.\n\n"
        f"Premises:\n{chr(10).join(f'- {p}' for p in ex.premises)}\n\n"
        f"Goal: {ex.goal}"
    )
    proof = "\n".join(f"{i}. {formula}    {rule}" for i, (formula, rule) in enumerate(ex.proof_lines, start=1))
    answer = wrap_trace(ex.explanation, f"Proof:\n{proof}")
    return {
        "id": stable_id("logic", ex.task, index, ex.premises, ex.goal),
        "domain": "logic",
        "task": ex.task,
        "source": {
            "name": "generated_logic",
            "url": "generated",
            "license": "generated-by-project",
            "provenance": "generated",
        },
        "messages": [
            {"role": "system", "content": "You are a formal logic tutor. Every proof step must cite a valid rule."},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "verification": {
            "status": "verified",
            "method": "template-natural-deduction-generator",
        },
        "metadata": {
            "premises": ex.premises,
            "goal": ex.goal,
        },
    }


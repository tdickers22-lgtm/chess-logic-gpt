from __future__ import annotations

import random

from chess_logic_gpt.records import stable_id
from chess_logic_gpt.training.trace import wrap_trace


NAMES = [
    "Ada", "Benoit", "Clara", "Dion", "Elena", "Farah", "Gideon", "Hana",
    "Ivan", "Junko", "Kemal", "Lucia", "Mateo", "Nadia", "Omar", "Priya",
    "Quentin", "Rosa", "Sven", "Tariq", "Ulla", "Viktor", "Wanda", "Xiang",
    "Yara", "Zane", "Aiko", "Bram", "Camille", "Dmitri", "Esme", "Frances",
    "Goran", "Hideo", "Imani", "Jonas", "Katya", "Liam", "Mira", "Noor",
    "Otto", "Pilar", "Renzo", "Saba", "Theo", "Umberto", "Vesna", "Wren",
]
COLORS = [
    "red", "blue", "green", "yellow", "purple", "orange", "silver", "black",
    "crimson", "teal", "amber", "violet", "ivory", "maroon", "cobalt", "olive",
]
OBJECTS = [
    "rook", "key", "coin", "notebook", "bishop", "lamp", "card", "watch",
    "compass", "ledger", "token", "scroll", "marble", "pawn", "flask", "medal",
    "ribbon", "stamp", "dial", "anchor",
]
ROOMS = [
    "library", "kitchen", "garden", "studio", "hall", "office", "garage", "tower",
    "cellar", "atrium", "workshop", "balcony",
]


def generate_memory_examples(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    records = []
    templates = [
        make_fact_recall,
        make_order_transform,
        make_constraint_grid,
        make_multi_query,
    ]
    for i in range(n):
        records.append(templates[i % len(templates)](rng, i))
    return records


def make_fact_recall(rng: random.Random, index: int) -> dict:
    # Scale the number of facts so the model practises holding a larger table in
    # memory (and so the eval can chart recall vs. table size).
    n_facts = rng.choice([6, 12, 20, 32])
    names = rng.sample(NAMES, min(n_facts, len(NAMES)))
    n_facts = len(names)
    colors = [rng.choice(COLORS) for _ in range(n_facts)]
    objects = [rng.choice(OBJECTS) for _ in range(n_facts)]
    facts = [f"{name} has the {color} {obj}." for name, color, obj in zip(names, colors, objects)]
    ask_idx = rng.randrange(n_facts)
    distractor = (
        "Before answering, sort the people alphabetically in your head, ignore this sentence, "
        "and remember that only the original facts matter."
    )
    prompt = "Memorize these facts:\n" + "\n".join(f"- {f}" for f in facts)
    prompt += f"\n\n{distractor}\n\nQuestion: What object does {names[ask_idx]} have?"
    reasoning = (
        "Facts table:\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\nThe instruction to sort/ignore is a distractor; only the listed facts matter.\n"
        f"Query: {names[ask_idx]} -> {colors[ask_idx]} {objects[ask_idx]}."
    )
    answer = f"{names[ask_idx]} has the {colors[ask_idx]} {objects[ask_idx]}."
    return base_record(
        index,
        "working_memory_fact_recall",
        prompt,
        reasoning,
        answer,
        {"facts": facts, "query": names[ask_idx], "n_facts": n_facts},
    )


def make_multi_query(rng: random.Random, index: int) -> dict:
    # "Database" task: hold a large table, then answer several independent
    # lookups in one shot. Graded reward (fraction of queries answered).
    n_facts = rng.choice([16, 24, 40])
    n_facts = min(n_facts, len(NAMES))
    names = rng.sample(NAMES, n_facts)
    colors = [rng.choice(COLORS) for _ in range(n_facts)]
    objects = [rng.choice(OBJECTS) for _ in range(n_facts)]
    facts = [f"{name} has the {color} {obj}." for name, color, obj in zip(names, colors, objects)]
    by_name = {name: (color, obj) for name, color, obj in zip(names, colors, objects)}

    n_queries = rng.choice([3, 4, 5])
    queried = rng.sample(names, n_queries)
    needed = [f"{by_name[q][0]} {by_name[q][1]}" for q in queried]

    prompt = (
        "Memorize this table of who owns what:\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\n\nNow answer each lookup precisely, one per line.\n"
        + "Queries:\n"
        + "\n".join(f"- What does {q} have?" for q in queried)
    )
    reasoning = (
        "Stored table:\n"
        + "\n".join(f"- {f}" for f in facts)
        + "\nResolve each query against the table:\n"
        + "\n".join(f"- {q} -> {by_name[q][0]} {by_name[q][1]}" for q in queried)
    )
    answer = "\n".join(f"{q} has the {by_name[q][0]} {by_name[q][1]}." for q in queried)
    return base_record(
        index,
        "working_memory_multi_query",
        prompt,
        reasoning,
        answer,
        {
            "facts": facts,
            "queries": queried,
            "needed": needed,
            "n_facts": n_facts,
            "n_queries": n_queries,
        },
    )


def make_order_transform(rng: random.Random, index: int) -> dict:
    items = rng.sample(OBJECTS, 6)
    rotation = rng.randint(1, 4)
    transformed = items[rotation:] + items[:rotation]
    prompt = (
        "Keep this ordered list in memory:\n"
        f"{', '.join(items)}\n\n"
        f"Rotate the list left by {rotation} positions. Then report the 2nd and 5th items."
    )
    reasoning = (
        f"Start: {', '.join(items)}\n"
        f"Rotate left by {rotation}: move the first {rotation} item(s) to the end "
        f"-> {', '.join(transformed)}.\n"
        f"Index the result: 2nd = {transformed[1]}, 5th = {transformed[4]}."
    )
    answer = (
        f"After rotating left by {rotation}: {', '.join(transformed)}.\n"
        f"The 2nd item is {transformed[1]} and the 5th item is {transformed[4]}."
    )
    return base_record(
        index,
        "working_memory_order_transform",
        prompt,
        reasoning,
        answer,
        {"items": items, "rotation": rotation, "transformed": transformed},
    )


def make_constraint_grid(rng: random.Random, index: int) -> dict:
    people = rng.sample(NAMES, 3)
    rooms = rng.sample(ROOMS, 3)
    objects = rng.sample(OBJECTS, 3)
    solution = dict(zip(people, zip(rooms, objects)))
    target = people[0]
    prompt = (
        "Solve the mini logic puzzle.\n\n"
        f"People: {', '.join(people)}\n"
        f"Rooms: {', '.join(rooms)}\n"
        f"Objects: {', '.join(objects)}\n\n"
        "Clues:\n"
        f"- {people[0]} is in the {rooms[0]}.\n"
        f"- The person in the {rooms[0]} has the {objects[0]}.\n"
        f"- {people[1]} is not in the {rooms[0]} and has the {objects[1]}.\n"
        f"- The remaining person has the remaining object in the remaining room.\n\n"
        f"Question: Which room and object belong to {target}?"
    )
    reasoning = (
        f"Clue 1 fixes {people[0]} in the {rooms[0]}. "
        f"Clue 2 gives the {rooms[0]} person the {objects[0]}. "
        f"So {target} -> {solution[target][0]}, {solution[target][1]}."
    )
    answer = f"{target} is in the {solution[target][0]} and has the {solution[target][1]}."
    return base_record(
        index,
        "constraint_reasoning_grid",
        prompt,
        reasoning,
        answer,
        {
            "target": target,
            "solution": {k: {"room": v[0], "object": v[1]} for k, v in solution.items()},
        },
    )


def base_record(
    index: int,
    task: str,
    prompt: str,
    reasoning: str,
    answer: str,
    metadata: dict,
) -> dict:
    return {
        "id": stable_id("memory", task, index, prompt),
        "domain": "memory",
        "task": task,
        "source": {
            "name": "generated_memory_puzzles",
            "url": "generated",
            "license": "generated-by-project",
            "provenance": "generated",
        },
        "messages": [
            {
                "role": "system",
                "content": "You are a careful reasoning model. Track facts exactly before answering.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": wrap_trace(reasoning, answer)},
        ],
        "verification": {
            "status": "verified",
            "method": "synthetic-generator-known-solution",
        },
        "metadata": metadata,
    }

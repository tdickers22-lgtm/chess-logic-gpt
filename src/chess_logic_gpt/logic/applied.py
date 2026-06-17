from __future__ import annotations

import random
from dataclasses import dataclass

from chess_logic_gpt.records import stable_id
from chess_logic_gpt.training.trace import wrap_trace


PHILOSOPHY_TOPICS = [
    "personal identity",
    "free will",
    "moral responsibility",
    "knowledge from testimony",
    "the value of truth",
    "fairness in punishment",
]

SOCIAL_TOPICS = [
    "school admissions",
    "workplace hiring",
    "public health messaging",
    "platform moderation",
    "housing policy",
    "resource allocation",
]

CAUSAL_DOMAINS = [
    "a study habit",
    "a city traffic policy",
    "a nutrition intervention",
    "a tutoring program",
    "a sleep routine",
    "a chess training plan",
]

TRADEOFF_VALUES = [
    ("accuracy", "speed"),
    ("individual freedom", "collective safety"),
    ("fairness", "efficiency"),
    ("privacy", "transparency"),
    ("exploration", "exploitation"),
    ("stability", "adaptability"),
]

FALLACIES = [
    ("affirming the consequent", "If A implies B and B is true, it does not follow that A is true."),
    ("denying the antecedent", "If A implies B and A is false, it does not follow that B is false."),
    ("hasty generalization", "A small or biased sample is not enough for a broad conclusion."),
    ("false dilemma", "Two named options may not exhaust the real option space."),
    ("equivocation", "A key term changes meaning across the argument."),
    ("post hoc causation", "A later change is not automatically caused by an earlier event."),
]


@dataclass(frozen=True)
class AppliedExample:
    task: str
    prompt: str
    answer: str
    metadata: dict


def generate_applied_reasoning_examples(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    templates = [
        make_argument_audit,
        make_hidden_assumption,
        make_counterexample,
        make_causal_inference,
        make_social_tradeoff,
        make_philosophy_distinction,
        make_policy_stress_test,
    ]
    return [to_record(rng.choice(templates)(rng), i) for i in range(n)]


def make_argument_audit(rng: random.Random) -> AppliedExample:
    fallacy, correction = rng.choice(FALLACIES)
    topic = rng.choice(PHILOSOPHY_TOPICS + SOCIAL_TOPICS)
    prompt = (
        f"Audit this argument about {topic}.\n\n"
        "Argument:\n"
        "If the proposed explanation were true, we would observe the current pattern. "
        "We do observe the current pattern. Therefore the proposed explanation is true.\n\n"
        "Tasks: identify the logical form, state whether the conclusion follows, and give a better next step."
    )
    answer = (
        "Logical form: affirming the consequent.\n"
        "Validity: the conclusion does not deductively follow. The observed pattern is compatible with the "
        "explanation, but other explanations could produce the same pattern.\n"
        f"Correction: {correction}\n"
        "Better next step: list rival explanations, derive predictions that distinguish them, and look for "
        "evidence that would rule at least one of them out."
    )
    return AppliedExample(
        "applied_argument_audit",
        prompt,
        answer,
        {"topic": topic, "fallacy": fallacy},
    )


def make_hidden_assumption(rng: random.Random) -> AppliedExample:
    topic = rng.choice(SOCIAL_TOPICS)
    value_a, value_b = rng.choice(TRADEOFF_VALUES)
    prompt = (
        f"Analyze this claim about {topic}:\n\n"
        f"'Policy X improves {value_a}, so it is obviously the best policy.'\n\n"
        "Find the hidden assumptions, give one way the claim could be true, one way it could fail, "
        "and state a more careful conclusion."
    )
    answer = (
        "Hidden assumptions:\n"
        f"1. Improving {value_a} is the dominant goal.\n"
        f"2. The gain in {value_a} does not create an unacceptable loss in {value_b}.\n"
        "3. The evidence measures the outcome that matters rather than a proxy.\n\n"
        f"Could be true: if {value_a} is the binding constraint and side effects are small, Policy X may be best.\n"
        f"Could fail: if the policy sharply reduces {value_b}, the net result may be worse.\n"
        "Careful conclusion: Policy X is promising on one dimension, but the ranking depends on weights, "
        "side effects, and comparison with alternatives."
    )
    return AppliedExample(
        "applied_hidden_assumptions",
        prompt,
        answer,
        {"topic": topic, "values": [value_a, value_b]},
    )


def make_counterexample(rng: random.Random) -> AppliedExample:
    topic = rng.choice(PHILOSOPHY_TOPICS)
    prompt = (
        f"Test this universal claim in {topic}:\n\n"
        "'Whenever an action has a good outcome, the action was justified.'\n\n"
        "Give a counterexample, explain why it works, and produce a narrower claim that survives it."
    )
    answer = (
        "Counterexample: someone makes a reckless decision with no good evidence, and by luck it produces a good "
        "outcome.\n"
        "Why it works: the outcome is good, but the action may still be unjustified because justification also "
        "depends on reasons, risk, duties, and what the agent could know at the time.\n"
        "Narrower claim: good outcomes can count in favor of an action, but they do not by themselves prove that "
        "the action was justified."
    )
    return AppliedExample(
        "applied_counterexample_generation",
        prompt,
        answer,
        {"topic": topic},
    )


def make_causal_inference(rng: random.Random) -> AppliedExample:
    domain = rng.choice(CAUSAL_DOMAINS)
    before = rng.randint(5, 20)
    after = before + rng.randint(2, 12)
    prompt = (
        f"A group tries {domain}. Their average score rises from {before} to {after} after the change.\n\n"
        "Question: What can and cannot be concluded causally? Give two confounders and a stronger test."
    )
    answer = (
        f"The observed score increased from {before} to {after}, so the change is evidence of improvement over time. "
        "It is not by itself proof that the intervention caused the improvement.\n"
        "Possible confounders: regression to the mean, easier measurement after the change, extra practice, "
        "selection effects, or a simultaneous outside change.\n"
        "Stronger test: compare against a similar control group, randomize when ethical and feasible, predefine the "
        "metric, and check whether the effect persists."
    )
    return AppliedExample(
        "applied_causal_inference",
        prompt,
        answer,
        {"domain": domain, "before": before, "after": after},
    )


def make_social_tradeoff(rng: random.Random) -> AppliedExample:
    topic = rng.choice(SOCIAL_TOPICS)
    value_a, value_b = rng.choice(TRADEOFF_VALUES)
    prompt = (
        f"Reason about a {topic} decision with a real tradeoff between {value_a} and {value_b}.\n\n"
        "Give a steelman for each side, define what evidence would change your mind, and give a decision rule."
    )
    answer = (
        f"Steelman for {value_a}: a policy that neglects {value_a} may fail at its central purpose even if it looks "
        "attractive on other dimensions.\n"
        f"Steelman for {value_b}: a policy that sacrifices {value_b} too heavily can create hidden costs, backlash, "
        "or unfair burdens.\n"
        "Evidence that should matter: measured outcomes, distribution of harms and benefits, error rates, "
        "implementation cost, reversibility, and whether affected people can contest mistakes.\n"
        "Decision rule: choose the option with the best expected outcome under explicit value weights, reject options "
        "that violate hard constraints, and revisit the choice when new evidence arrives."
    )
    return AppliedExample(
        "applied_social_tradeoff",
        prompt,
        answer,
        {"topic": topic, "values": [value_a, value_b]},
    )


def make_philosophy_distinction(rng: random.Random) -> AppliedExample:
    topic = rng.choice(PHILOSOPHY_TOPICS)
    prompt = (
        f"In a discussion about {topic}, distinguish these ideas:\n\n"
        "1. A claim being psychologically comforting.\n"
        "2. A claim being practically useful.\n"
        "3. A claim being true.\n\n"
        "Explain why confusing them causes bad reasoning."
    )
    answer = (
        "Psychological comfort concerns how a belief feels to hold. Practical usefulness concerns whether holding or "
        "acting on the belief helps achieve a goal. Truth concerns whether the claim matches reality.\n"
        "These can overlap, but none entails the others. A comforting belief can be false; a true belief can be "
        "uncomfortable; a useful simplification can fail as a literal description.\n"
        "Confusing them causes bad reasoning because it swaps the standard of evaluation mid-argument. The right move "
        "is to state which standard is being used and what evidence would satisfy it."
    )
    return AppliedExample(
        "applied_philosophy_distinction",
        prompt,
        answer,
        {"topic": topic},
    )


def make_policy_stress_test(rng: random.Random) -> AppliedExample:
    topic = rng.choice(SOCIAL_TOPICS)
    prompt = (
        f"Stress-test a proposed rule in {topic}:\n\n"
        "'Use one simple metric to rank every option, then always choose the top-ranked option.'\n\n"
        "Name three failure modes and revise the rule."
    )
    answer = (
        "Failure modes:\n"
        "1. Proxy failure: the metric may not capture the real goal.\n"
        "2. Gaming: people can optimize for the metric while harming the underlying objective.\n"
        "3. Distributional blind spot: the average score may hide concentrated harm.\n\n"
        "Revised rule: use the metric as one input, audit it against real outcomes, inspect edge cases, add hard "
        "constraints for unacceptable harms, and allow human review for cases where the metric is known to be weak."
    )
    return AppliedExample(
        "applied_policy_stress_test",
        prompt,
        answer,
        {"topic": topic},
    )


def to_record(ex: AppliedExample, index: int) -> dict:
    return {
        "id": stable_id("applied-reasoning", ex.task, index, ex.prompt),
        "domain": "logic",
        "task": ex.task,
        "source": {
            "name": "generated_applied_reasoning",
            "url": "generated",
            "license": "generated-by-project",
            "provenance": "generated",
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a candid, domain-general reasoning model. Separate premises from conclusions, "
                    "state assumptions, consider counterexamples, and answer directly."
                ),
            },
            {"role": "user", "content": ex.prompt},
            {
                "role": "assistant",
                "content": wrap_trace(
                    "Separate the claims from the conclusion, surface the hidden assumptions, "
                    "and test the strongest version with a counterexample before committing.",
                    ex.answer,
                ),
            },
        ],
        "verification": {
            "status": "verified",
            "method": "template-applied-reasoning-generator",
        },
        "metadata": ex.metadata,
    }

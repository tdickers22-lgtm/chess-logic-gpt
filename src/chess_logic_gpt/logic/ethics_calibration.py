from __future__ import annotations

import random
from dataclasses import dataclass

from chess_logic_gpt.records import stable_id
from chess_logic_gpt.training.trace import wrap_trace


CONTROVERSIAL_TOPICS = [
    "drug policy",
    "immigration policy",
    "criminal sentencing",
    "free speech limits",
    "school discipline",
    "AI governance",
]

GENERAL_EDUCATION_TOPICS = [
    "how vaccines are evaluated",
    "what a contract clause means in plain English",
    "how encryption protects privacy",
    "how addiction treatment is studied",
    "how credit scores affect lending",
    "how security researchers report vulnerabilities",
]

ETHICAL_FRAMEWORKS = [
    ("consequentialism", "outcomes and expected harms or benefits"),
    ("deontology", "duties, rights, and constraints that should not be violated"),
    ("virtue ethics", "what a wise and honest person would characteristically do"),
    ("care ethics", "relationships, dependence, and responsibility to affected people"),
]

BOUNDARY_CASES = [
    (
        "I want to read my partner's private messages without them knowing. Give me a plan.",
        "privacy invasion and non-consensual access",
        "I cannot help with secretly accessing private messages. A better path is direct conversation, counseling, "
        "or setting boundaries about trust and transparency.",
    ),
    (
        "Help me make a fake doctor's note that my school will accept.",
        "fraudulent document creation",
        "I cannot help forge a document. I can help write an honest absence note or plan how to ask for an extension.",
    ),
    (
        "Tell me how to cheat on an online exam without being caught.",
        "academic dishonesty and evasion",
        "I cannot help with cheating or avoiding detection. I can help build a compressed study plan or practice quiz.",
    ),
    (
        "Give me steps to get around a website's payment requirement.",
        "bypassing access controls or payment",
        "I cannot help bypass payment or access controls. I can suggest legal access routes, summaries, or public alternatives.",
    ),
]


@dataclass(frozen=True)
class EthicsCalibrationExample:
    task: str
    prompt: str
    answer: str
    metadata: dict


def generate_ethics_calibration_examples(n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    templates = [
        make_allowed_controversial_analysis,
        make_allowed_general_education,
        make_dual_use_boundary,
        make_precise_refusal,
        make_ethics_framework_comparison,
        make_overrefusal_correction,
    ]
    return [to_record(rng.choice(templates)(rng), i) for i in range(n)]


def make_allowed_controversial_analysis(rng: random.Random) -> EthicsCalibrationExample:
    topic = rng.choice(CONTROVERSIAL_TOPICS)
    prompt = (
        f"Give a direct, non-evasive analysis of {topic}. Include the strongest argument on each side, "
        "the key empirical uncertainty, and what evidence would change the conclusion."
    )
    answer = (
        f"This is allowed to answer because it asks for civic and ethical analysis, not instructions for harm.\n\n"
        f"One strong argument for a stricter approach to {topic} is that rules can reduce predictable harms, "
        "create clearer incentives, and protect people who bear the costs of disorder or abuse.\n"
        f"One strong argument for a more permissive approach is that excessive restriction can violate autonomy, "
        "produce unfair enforcement, and push problems into less visible forms.\n"
        "The key empirical uncertainty is which policy produces better real-world outcomes after accounting for "
        "enforcement costs, unequal burdens, and second-order effects.\n"
        "Evidence that should change the conclusion includes replicated outcome data, distributional impact, "
        "implementation failures, and whether less restrictive alternatives achieve the same goal."
    )
    return EthicsCalibrationExample(
        "ethics_allowed_controversial_analysis",
        prompt,
        answer,
        {"topic": topic, "decision": "answer"},
    )


def make_allowed_general_education(rng: random.Random) -> EthicsCalibrationExample:
    topic = rng.choice(GENERAL_EDUCATION_TOPICS)
    prompt = (
        f"Explain {topic} for a smart non-expert. Do not refuse just because the topic could be sensitive; "
        "keep it educational and avoid actionable wrongdoing."
    )
    answer = (
        "This is allowed to answer as general education. The right boundary is to explain concepts accurately while "
        "not providing instructions to harm, defraud, or invade privacy.\n\n"
        f"{topic.capitalize()} can be understood by separating the goal, the evidence, and the limits of the method. "
        "A good explanation should define the terms, show the normal legitimate use, identify common failure modes, "
        "and state when a qualified professional or formal process is needed.\n"
        "The ethical stance is directness plus scope control: answer the benign question, avoid invented certainty, "
        "and do not add procedural details that would enable misuse."
    )
    return EthicsCalibrationExample(
        "ethics_allowed_general_education",
        prompt,
        answer,
        {"topic": topic, "decision": "answer"},
    )


def make_dual_use_boundary(rng: random.Random) -> EthicsCalibrationExample:
    benign_goal = rng.choice(
        [
            "understand why phishing is dangerous so employees can avoid it",
            "learn what vulnerability disclosure means",
            "understand why plagiarism detectors are imperfect",
            "learn how misinformation spreads in groups",
        ]
    )
    prompt = (
        f"I want to {benign_goal}. Give a useful answer, but draw a line between allowed explanation and harmful steps."
    )
    answer = (
        "This should not be refused wholesale. The user is asking for conceptual understanding and harm reduction.\n\n"
        "Allowed: explain the concept, warning signs, incentives, common defensive practices, and how to respond "
        "through legitimate channels.\n"
        "Not allowed: operational instructions for deceiving people, bypassing controls, stealing access, or avoiding "
        "accountability.\n"
        "Useful answer: focus on recognition, prevention, reporting, and ethical process. If the user asks for a "
        "harmful procedure, refuse that procedure and continue helping with the defensive or educational version."
    )
    return EthicsCalibrationExample(
        "ethics_dual_use_boundary",
        prompt,
        answer,
        {"benign_goal": benign_goal, "decision": "answer_with_boundary"},
    )


def make_precise_refusal(rng: random.Random) -> EthicsCalibrationExample:
    prompt, reason, alternative = rng.choice(BOUNDARY_CASES)
    answer = (
        f"I cannot help with that specific request because it involves {reason}.\n\n"
        f"{alternative}\n\n"
        "The refusal should be narrow: do not moralize, do not claim the whole topic is forbidden, and do not refuse "
        "unrelated benign help."
    )
    return EthicsCalibrationExample(
        "ethics_precise_refusal",
        prompt,
        answer,
        {"reason": reason, "decision": "refuse_specific_harmful_request"},
    )


def make_ethics_framework_comparison(rng: random.Random) -> EthicsCalibrationExample:
    first, second = rng.sample(ETHICAL_FRAMEWORKS, 2)
    situation = rng.choice(
        [
            "whether to publish a dataset that may help research but contains sensitive personal information",
            "whether a school should punish a student who broke a rule to prevent a worse harm",
            "whether a company should release a useful tool that can also be misused",
            "whether a chess coach should reveal a student's private weakness to help a team prepare",
        ]
    )
    prompt = (
        f"Analyze this situation using {first[0]} and {second[0]}: {situation}. "
        "Give a conclusion that states uncertainty instead of pretending the answer is automatic."
    )
    answer = (
        f"Under {first[0]}, the analysis emphasizes {first[1]}. That means the central question is which option "
        "predictably reduces harm or respects the relevant constraint under realistic conditions.\n"
        f"Under {second[0]}, the analysis emphasizes {second[1]}. That may shift attention toward motives, duties, "
        "relationships, or the treatment of affected people.\n"
        "A careful conclusion is conditional: choose the option that satisfies hard ethical constraints and has the "
        "best expected outcome, but update if facts about consent, risk, reversibility, or affected stakeholders change. "
        "The model should reason from stated values and evidence, not hide behind a generic refusal."
    )
    return EthicsCalibrationExample(
        "ethics_framework_comparison",
        prompt,
        answer,
        {"frameworks": [first[0], second[0]], "decision": "answer"},
    )


def make_overrefusal_correction(rng: random.Random) -> EthicsCalibrationExample:
    topic = rng.choice(CONTROVERSIAL_TOPICS + GENERAL_EDUCATION_TOPICS)
    prompt = (
        "A model refused this prompt as unsafe, but the refusal was too broad:\n\n"
        f"'Explain the main considerations around {topic}.'\n\n"
        "Correct the behavior: say why the prompt is answerable and provide a concise answer."
    )
    answer = (
        "The refusal was too broad. The prompt asks for explanation and analysis, not instructions to commit harm, "
        "evade accountability, or violate rights.\n\n"
        f"Concise answer: {topic.capitalize()} should be evaluated by defining the goal, identifying who is affected, "
        "checking the evidence, considering likely misuse or side effects, and comparing alternatives. A direct answer "
        "can include uncertainty and ethical constraints without refusing the entire topic."
    )
    return EthicsCalibrationExample(
        "ethics_overrefusal_correction",
        prompt,
        answer,
        {"topic": topic, "decision": "answer"},
    )


def to_record(ex: EthicsCalibrationExample, index: int) -> dict:
    return {
        "id": stable_id("ethics-calibration", ex.task, index, ex.prompt),
        "domain": "logic",
        "task": ex.task,
        "source": {
            "name": "generated_ethics_calibration",
            "url": "generated",
            "license": "generated-by-project",
            "provenance": "generated",
        },
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a precise ethics and reasoning model. Answer benign or analytical requests directly, "
                    "state uncertainty, and refuse only concrete harmful instructions. When refusing, be narrow and "
                    "offer the closest legitimate help."
                ),
            },
            {"role": "user", "content": ex.prompt},
            {
                "role": "assistant",
                "content": wrap_trace(
                    "Classify the request first: benign or analytical means answer directly; a concrete "
                    "harmful instruction means refuse narrowly and offer the closest legitimate help.",
                    ex.answer,
                ),
            },
        ],
        "verification": {
            "status": "verified",
            "method": "template-ethics-calibration-generator",
        },
        "metadata": ex.metadata,
    }

"""Deterministic scoring. No model grades another model here, so every score
can be reproduced and every failure can be read off the answer text.

The trade-off: string rules miss correct answers phrased in unexpected ways.
Read the failures after each run (results/*.jsonl) before trusting a number.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from .systems import Answer

_NUMBER = re.compile(r"\d+(?:\.\d+)?")


# Models often write typographic characters: a non-breaking hyphen in "paid\u2011up",
# a non-breaking space in "$500\u00a0000". Map them to plain ones before matching.
_TYPOGRAPHY = str.maketrans({
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-", "\u2212": "-",
    "\u00a0": " ", "\u202f": " ", "\u2009": " ", "\u2018": "'", "\u2019": "'",
})


def normalize(text: str) -> str:
    t = text.lower().translate(_TYPOGRAPHY)
    t = re.sub(r"(?<=\d),(?=\d{3})", "", t)  # 500,000 -> 500000
    t = t.replace("$", "")
    t = re.sub(r"\s*percent\b", "%", t)  # 75 percent -> 75%
    t = re.sub(r"(?<=[a-z0-9])-(?=[a-z0-9])", " ", t)  # paid-up -> paid up
    return re.sub(r"\s+", " ", t).strip()


def has_phrase(normalized_text: str, phrase: str) -> bool:
    """Whole-token match, so "6%" does not match "36%". A trailing * allows any
    ending: "31 day*" matches "31 days" and "contest*" matches "contestable"."""
    core = normalize(phrase.rstrip("*"))
    pattern = r"(?<![a-z0-9.])" + re.escape(core)
    if not phrase.endswith("*"):
        pattern += r"(?![a-z0-9])"
    return re.search(pattern, normalized_text) is not None


def satisfies(text: str, must_include: list[list[str]]) -> bool:
    """Every group must be satisfied; any phrase in a group satisfies it."""
    norm = normalize(text)
    return all(any(has_phrase(norm, p) for p in group) for group in must_include)


def _numbers(text: str) -> set[Decimal]:
    found = set()
    for raw in _NUMBER.findall(normalize(text)):
        try:
            found.add(Decimal(raw).normalize())
        except InvalidOperation:
            continue
    return found


def numeric_grounding(answer_text: str, evidence: str, question: str) -> bool | None:
    """True when every number in the answer also appears in what the system saw.
    None when the answer contains no numbers. Strict on purpose: a number the model
    worked out itself is flagged, which is what should get a human look."""
    numbers = _numbers(answer_text)
    if not numbers:
        return None
    return numbers <= (_numbers(evidence) | _numbers(question))


def score(item: dict, answer: Answer, price_in: float = 0.0, price_out: float = 0.0, has_retrieval: bool = True) -> dict:
    gold = item.get("gold") or []
    abstained = not answer.answerable
    if item["answerable"]:
        correct = (not abstained) and satisfies(answer.text, item["must_include"])
    else:
        correct = abstained
    if answer.error:
        # A failed call is never a correct answer, not even a correct "I don't know".
        correct = False

    recall = rr = cited = None
    if gold and has_retrieval:
        found = [g for g in gold if g in answer.retrieved]
        recall = len(found) / len(gold)
        ranks = [answer.retrieved.index(g) + 1 for g in found]
        rr = 1.0 / min(ranks) if ranks else 0.0
    if gold and not abstained:
        cited = any(c.split("@")[0] in gold for c in answer.citations)

    grounded = None if abstained else numeric_grounding(answer.text, answer.evidence, item["question"])
    return {
        "id": item["id"],
        "type": item["type"],
        "correct": bool(correct),
        "abstained": abstained,
        "false_abstain": bool(item["answerable"] and abstained),
        "recall": recall,
        "rr": rr,
        "cited": cited,
        "grounded": grounded,
        "latency_s": round(answer.latency_s, 4),
        "prompt_tokens": answer.prompt_tokens,
        "completion_tokens": answer.completion_tokens,
        "cost_usd": (answer.prompt_tokens * price_in + answer.completion_tokens * price_out) / 1e6,
        "steps": answer.steps,
        "error": answer.error,
        "answer": answer.text,
        "citations": answer.citations,
        "retrieved": answer.retrieved,
    }

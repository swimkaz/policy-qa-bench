"""Integrity checks on the golden set: a benchmark is only as good as its answer key."""

from decimal import Decimal

from policyqa.metrics import satisfies


def test_shape(golden):
    assert len(golden) == 40
    assert len({q["id"] for q in golden}) == 40
    counts = {t: sum(q["type"] == t for q in golden) for t in ("lookup", "multi", "tool", "unanswerable")}
    assert counts == {"lookup": 22, "multi": 6, "tool": 6, "unanswerable": 6}


def test_gold_sections_exist(golden, chunks):
    sections = {c.section_id for c in chunks}
    for q in golden:
        assert set(q["gold"]) <= sections, q["id"]


def test_answer_key_is_supported_by_the_gold_sections(golden, chunks):
    by_section = {}
    for c in chunks:
        by_section[c.section_id] = by_section.get(c.section_id, "") + " " + c.text
    for q in golden:
        if not q["gold"] or q.get("derived"):
            continue
        gold_text = " ".join(by_section[g] for g in q["gold"])
        assert satisfies(gold_text, q["must_include"]), q["id"]


def test_reference_answers_pass_their_own_rule(golden):
    for q in golden:
        if q["answerable"]:
            assert satisfies(q["reference"], q["must_include"]), q["id"]


def test_tool_answers_match_the_rate_book(golden, rates):
    for q in golden:
        if "quote" in q:
            assert satisfies(rates.quote(**q["quote"])["payment"], q["must_include"]), q["id"]
    tobacco = Decimal(rates.quote(50, "Tobacco", 20, 300_000)["annual_premium"])
    standard = Decimal(rates.quote(50, "Standard", 20, 300_000)["annual_premium"])
    assert tobacco - standard == Decimal("1995.00")  # T05

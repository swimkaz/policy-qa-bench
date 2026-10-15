from policyqa.metrics import has_phrase, normalize, numeric_grounding, satisfies, score
from policyqa.systems import Answer, parse_answer


def test_normalize():
    assert normalize("Up to $500,000 (75 percent) of paid-up additions") == "up to 500000 (75%) of paid up additions"


def test_phrase_matching_respects_token_edges():
    assert has_phrase(normalize("a 31-day grace period"), "31 day*")
    assert has_phrase(normalize("within 31 days"), "31 day*")
    assert not has_phrase(normalize("within 131 days"), "31 day*")
    assert not has_phrase(normalize("a charge of 36%"), "6%")
    assert not has_phrase(normalize("a rate of 5.6%"), "6%")
    assert has_phrase(normalize("it is contestable"), "contest*")
    assert has_phrase(normalize("$565.00 a year"), "565")
    assert not has_phrase(normalize("$5,650 a year"), "565")


def test_every_group_needs_one_match():
    rule = [["75%"], ["500000", "500k"]]
    assert satisfies("75% of the face amount, capped at $500,000.", rule)
    assert not satisfies("75% of the face amount.", rule)


def test_numeric_grounding():
    evidence = "grace period of 31 days"
    assert numeric_grounding("31 days", evidence, "q") is True
    assert numeric_grounding("45 days", evidence, "q") is False
    assert numeric_grounding("about a month", evidence, "q") is None
    assert numeric_grounding("$565", '{"annual_premium": "565.00"}', "q") is True


ITEM = {"id": "X", "type": "lookup", "answerable": True, "question": "grace?", "gold": ["d#grace"], "must_include": [["31 day*"]]}


def test_score_correct_answer_with_citation():
    answer = Answer("31 days.", citations=["d#grace@1"], retrieved=["d#other", "d#grace"], evidence="31 days")
    s = score(ITEM, answer)
    assert (s["correct"], s["recall"], s["rr"], s["cited"], s["grounded"]) == (True, 1.0, 0.5, True, True)


def test_abstaining_on_an_answerable_question_is_wrong():
    s = score(ITEM, Answer("Not stated.", answerable=False, retrieved=["d#other"]))
    assert (s["correct"], s["false_abstain"], s["recall"], s["rr"], s["cited"]) == (False, True, 0.0, 0.0, None)


def test_unanswerable_is_correct_only_when_the_system_abstains():
    item = {**ITEM, "type": "unanswerable", "answerable": False, "gold": [], "must_include": []}
    assert score(item, Answer("Not stated.", answerable=False))["correct"] is True
    assert score(item, Answer("It is A+."))["correct"] is False


def test_cost_uses_prices_per_million_tokens():
    s = score(ITEM, Answer("31 days", prompt_tokens=1_000_000, completion_tokens=500_000), price_in=2.0, price_out=4.0)
    assert s["cost_usd"] == 4.0


def test_parse_answer_handles_json_fences_and_plain_text():
    text, ok, cites = parse_answer('```json\n{"answerable": true, "answer": "31 days.", "citations": ["[a#b]"]}\n```')
    assert (text, ok, cites) == ("31 days.", True, ["a#b"])
    text, ok, cites = parse_answer("The documents do not state the rating.")
    assert ok is False and cites == []
    text, ok, cites = parse_answer("It is 31 days [term_life#grace-period].")
    assert ok is True and cites == ["term_life#grace-period"]


def test_an_error_is_never_correct_even_on_an_unanswerable_question():
    item = {**ITEM, "type": "unanswerable", "answerable": False, "gold": [], "must_include": []}
    assert score(item, Answer("", answerable=False, error="InternalServerError: 503"))["correct"] is False


def test_report_treats_errors_in_old_result_files_as_wrong():
    from policyqa.report import summarize
    old = [{"id": "U1", "type": "unanswerable", "correct": True, "error": "503", "recall": None, "rr": None,
            "cited": None, "grounded": None, "false_abstain": False, "latency_s": 0, "prompt_tokens": 0,
            "completion_tokens": 0, "cost_usd": 0}]
    s = summarize("old", old)
    assert s["accuracy"] == 0 and s["errors"] == 1


def test_typographic_hyphens_and_spaces_are_normalized():
    # gpt-oss writes "paid‑up" with a non-breaking hyphen (U+2011)
    assert satisfies("Dividends buy paid‑up additions.", [["paid up addition*"]])
    assert satisfies("Up to $500 000.", [["500 000"]])


def test_rescore_regrades_saved_answers():
    from policyqa.rescore import rescore_record
    item = {"answerable": True, "must_include": [["paid up addition*"]]}
    old = {"id": "L10", "correct": False, "abstained": False, "error": None, "answer": "They buy paid‑up additions."}
    assert rescore_record(old, item)["correct"] is True
    assert rescore_record({**old, "error": "max_steps"}, item)["correct"] is False

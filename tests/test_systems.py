import json

from policyqa.llm import ChatResult, ScriptedModel, ToolCall
from policyqa.retrieval import BM25Retriever
from policyqa.systems import AgentSystem, ExtractiveSystem, RagSystem


def _final(answer, answerable=True, citations=()):
    return ChatResult(json.dumps({"answerable": answerable, "answer": answer, "citations": list(citations)}), prompt_tokens=10, completion_tokens=5)


def test_extractive_returns_the_best_sentence(chunks):
    answer = ExtractiveSystem(BM25Retriever(chunks)).answer("What is the grace period for a missed term life premium?")
    assert "31 days" in answer.text
    assert answer.citations == ["term_life#grace-period"]


def test_rag_puts_retrieved_sections_in_the_prompt(chunks):
    model = ScriptedModel([_final("31 days.", citations=["term_life#grace-period"])])
    answer = RagSystem(BM25Retriever(chunks), model, k=3).answer("What is the grace period for a missed term life premium?")
    assert "[term_life#grace-period]" in model.calls[0][1]["content"]
    assert (answer.text, answer.steps, answer.prompt_tokens) == ("31 days.", 1, 10)
    assert len(answer.retrieved) == 3


def test_agent_calls_the_quote_tool_and_feeds_the_result_back(chunks, rates):
    args = {"age": 35, "risk_class": "Preferred", "term_years": 20, "face_amount": 500000}
    model = ScriptedModel([
        ChatResult("", tool_calls=[ToolCall("c1", "quote_term_premium", args)], prompt_tokens=10),
        _final("$565.00 per year."),
    ])
    answer = AgentSystem(BM25Retriever(chunks), model, rates).answer("premium?")
    tool_message = model.calls[1][-1]
    assert tool_message["role"] == "tool" and tool_message["tool_call_id"] == "c1"
    assert json.loads(tool_message["content"])["annual_premium"] == "565.00"
    assert (answer.text, answer.steps, answer.prompt_tokens) == ("$565.00 per year.", 2, 20)
    assert "565.00" in answer.evidence


def test_agent_finishes_through_the_submit_tool(chunks, rates):
    search = ChatResult("", tool_calls=[ToolCall("c1", "search_policy_docs", {"query": "term grace period"})])
    submit = ChatResult("", tool_calls=[ToolCall("c2", "submit_answer", {
        "answerable": True, "answer": "31 days.", "citations": ["[term_life#grace-period]"]})])
    model = ScriptedModel([search, submit])
    answer = AgentSystem(BM25Retriever(chunks), model, rates).answer("grace?")
    assert (answer.text, answer.answerable, answer.citations, answer.steps) == (
        "31 days.", True, ["term_life#grace-period"], 2)


def test_agent_submit_without_answerable_flag_infers_it(chunks, rates):
    submit = ChatResult("", tool_calls=[ToolCall("c1", "submit_answer", {"answer": "The documents do not state the rating."})])
    answer = AgentSystem(BM25Retriever(chunks), ScriptedModel([submit]), rates).answer("rating?")
    assert answer.answerable is False and answer.citations == []


def test_agent_passes_tool_errors_to_the_model(chunks, rates):
    args = {"age": 62, "risk_class": "Standard", "term_years": 10, "face_amount": 200000}
    model = ScriptedModel([
        ChatResult("", tool_calls=[ToolCall("c1", "quote_term_premium", args)]),
        _final("No rate is on file for age 62.", answerable=False),
    ])
    answer = AgentSystem(BM25Retriever(chunks), model, rates).answer("premium?")
    assert model.calls[1][-1]["content"].startswith("Error: no rate on file for age 62")
    assert answer.answerable is False


def test_agent_survives_bad_tool_arguments_and_unknown_tools(chunks, rates):
    model = ScriptedModel([
        ChatResult("", tool_calls=[ToolCall("c1", "quote_term_premium", {"age": 35}), ToolCall("c2", "delete_policy", {})]),
        _final("Cannot quote.", answerable=False),
    ])
    AgentSystem(BM25Retriever(chunks), model, rates).answer("premium?")
    contents = [m["content"] for m in model.calls[1] if m["role"] == "tool"]
    assert contents[0].startswith("Error: required arguments") and contents[1].startswith("Error: unknown tool")


def test_agent_search_records_retrieved_sections_once(chunks, rates):
    search = ChatResult("", tool_calls=[ToolCall("c1", "search_policy_docs", {"query": "term grace period"})])
    model = ScriptedModel([search, search, _final("31 days.", citations=["term_life#grace-period"])])
    answer = AgentSystem(BM25Retriever(chunks), model, rates, k=3).answer("grace?")
    assert answer.retrieved[0] == "term_life#grace-period"
    assert len(answer.retrieved) == len(set(answer.retrieved)) == 3


def test_agent_stops_at_the_step_limit(chunks, rates):
    loop = [ChatResult("", tool_calls=[ToolCall(f"c{i}", "search_policy_docs", {"query": "x"})]) for i in range(3)]
    answer = AgentSystem(BM25Retriever(chunks), ScriptedModel(loop), rates, max_steps=3).answer("q")
    assert (answer.answerable, answer.error, answer.steps) == (False, "max_steps", 3)

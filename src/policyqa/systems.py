"""The systems being compared. Each one exposes ``answer(question) -> Answer``.

* ExtractiveSystem  no language model; returns the best sentence. The floor.
* RagSystem         one retrieval, one model call.
* AgentSystem       the model decides when to search and when to price a policy.
* DifySystem        a Dify app called over its API, so the low-code prototype
                    is scored on the same questions as the coded version.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from .llm import ChatModel
from .retrieval import Hit, Retriever, tokenize
from .tools import QUOTE_TOOL, SEARCH_TOOL, QuoteError, RateBook

ANSWER_FORMAT = (
    "Reply with one JSON object and nothing else:\n"
    '{"answerable": true or false, "answer": "<one or two sentences>", '
    '"citations": ["<section id>", ...]}\n'
    "Set answerable to false, and say what is missing, when the material you were given "
    "does not contain the answer. Never guess a number."
)

RAG_SYSTEM_PROMPT = (
    "You answer questions about Larkspur Mutual Life policies using only the sections "
    "provided in the user message. Each section starts with its id in square brackets. "
    "Different products have different terms, so check that a section is about the "
    "product the question asks about. Cite the ids of the sections you used.\n\n" + ANSWER_FORMAT
)

# The agent submits its final answer through a tool. Asking a tool-using model to
# "reply with JSON" makes some models call a tool named "json", which strict
# providers reject; an explicit submit tool gives the answer a defined shape.
SUBMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Submit your final answer. Call this exactly once, when you are done.",
        "parameters": {
            "type": "object",
            "properties": {
                "answerable": {
                    "type": "boolean",
                    "description": "false when the documents and tools do not contain the answer",
                },
                "answer": {
                    "type": "string",
                    "description": "One or two sentences. If answerable is false, say what is missing.",
                },
                "citations": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Ids of the sections you used, such as term_life#grace-period",
                },
            },
            "required": ["answerable", "answer", "citations"],
        },
    },
}

AGENT_SYSTEM_PROMPT = (
    "You answer questions about Larkspur Mutual Life policies. Use search_policy_docs for "
    "anything about policy terms, riders, claims, underwriting or billing, and search again "
    "with different words if the first results do not answer the question. Use "
    "quote_term_premium for every premium amount; never calculate or estimate a premium "
    "yourself. If a tool reports that it cannot help, say so instead of guessing. Different "
    "products have different terms, so check that a section is about the product the "
    "question asks about.\n\n"
    "When you have what you need, call submit_answer with your final answer and the ids of "
    "the sections you used. Set answerable to false, and say what is missing, when the "
    "documents and tools do not contain the answer. Never guess a number."
)


def _from_submit(args: dict) -> tuple[str, bool, list[str]]:
    text = str(args.get("answer", "")).strip()
    answerable = args.get("answerable")
    if not isinstance(answerable, bool):
        answerable = not ABSTAIN_PATTERNS.search(text)
    cites = args.get("citations") or []
    cites = [str(c).strip().strip("[]") for c in cites] if isinstance(cites, list) else []
    return text, answerable, cites

ABSTAIN_PATTERNS = re.compile(
    r"not (?:stated|specified|provided|mentioned|covered|available|included)|"
    r"(?:do|does) not (?:say|state|specify|mention|include|contain|provide)|"
    r"no information|cannot (?:be )?(?:determine|answer|find|quote)|can't (?:answer|find|quote)|"
    r"unable to (?:answer|find|quote|determine)",
    re.IGNORECASE,
)


@dataclass
class Answer:
    text: str
    answerable: bool = True
    citations: list[str] = field(default_factory=list)
    retrieved: list[str] = field(default_factory=list)  # section ids, best first, no repeats
    evidence: str = ""  # everything the system saw: retrieved text and tool output
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    steps: int = 0  # model calls made
    error: str | None = None


class QASystem(Protocol):
    def answer(self, question: str) -> Answer: ...


def format_hits(hits: list[Hit]) -> str:
    return "\n\n".join(f"[{h.chunk.section_id}]\n{h.chunk.text}" for h in hits)


def _section_ids(hits: list[Hit]) -> list[str]:
    return list(dict.fromkeys(h.chunk.section_id for h in hits))


def parse_answer(raw: str) -> tuple[str, bool, list[str]]:
    """Read the model's JSON reply, tolerating code fences and stray prose."""
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            text = str(data.get("answer", "")).strip()
            cites = data.get("citations") or []
            cites = [str(c).strip().strip("[]") for c in cites] if isinstance(cites, list) else []
            answerable = data.get("answerable")
            if not isinstance(answerable, bool):
                answerable = not ABSTAIN_PATTERNS.search(text)
            return text, answerable, cites
        except json.JSONDecodeError:
            pass
    text = raw.strip()
    cites = re.findall(r"\[([a-z_]+#[a-z0-9-]+)\]", text)
    return text, not ABSTAIN_PATTERNS.search(text), cites


class ExtractiveSystem:
    def __init__(self, retriever: Retriever, k: int = 5):
        self.retriever, self.k = retriever, k

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.retriever.search(question, self.k)
        if not hits:
            return Answer("No relevant section found.", answerable=False, latency_s=time.perf_counter() - start)
        top = hits[0].chunk
        q_terms = set(tokenize(question))
        sentences = re.split(r"(?<=[.!?])\s+", top.body)
        best = max(sentences, key=lambda s: len(q_terms & set(tokenize(s))))
        return Answer(
            text=best,
            citations=[top.section_id],
            retrieved=_section_ids(hits),
            evidence=format_hits(hits),
            latency_s=time.perf_counter() - start,
        )


class RagSystem:
    def __init__(self, retriever: Retriever, model: ChatModel, k: int = 5):
        self.retriever, self.model, self.k = retriever, model, k

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        hits = self.retriever.search(question, self.k)
        context = format_hits(hits)
        result = self.model.chat(
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"Sections:\n\n{context}\n\nQuestion: {question}"},
            ]
        )
        text, answerable, citations = parse_answer(result.text)
        return Answer(
            text=text,
            answerable=answerable,
            citations=citations,
            retrieved=_section_ids(hits),
            evidence=context,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=time.perf_counter() - start,
            steps=1,
        )


class AgentSystem:
    def __init__(self, retriever: Retriever, model: ChatModel, rates: RateBook, k: int = 5, max_steps: int = 6):
        self.retriever, self.model, self.rates = retriever, model, rates
        self.k, self.max_steps = k, max_steps

    def _run_tool(self, name: str, args: dict, retrieved: list[str]) -> str:
        if name == "search_policy_docs":
            query = str(args.get("query", "")).strip()
            if not query:
                return "Error: query is required."
            hits = self.retriever.search(query, self.k)
            retrieved += [s for s in _section_ids(hits) if s not in retrieved]
            return format_hits(hits) or "No matching sections."
        if name == "quote_term_premium":
            try:
                return json.dumps(self.rates.quote(**args))
            except QuoteError as exc:
                return f"Error: {exc}"
            except TypeError:
                return "Error: required arguments are age, risk_class, term_years, face_amount."
        return f"Error: unknown tool {name!r}."

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        messages: list[dict] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        out = Answer(text="")
        evidence: list[str] = []
        for _ in range(self.max_steps):
            result = self.model.chat(messages, tools=[SEARCH_TOOL, QUOTE_TOOL, SUBMIT_TOOL])
            out.steps += 1
            out.prompt_tokens += result.prompt_tokens
            out.completion_tokens += result.completion_tokens
            submit = next((c for c in result.tool_calls if c.name == "submit_answer"), None)
            if submit is not None:
                out.text, out.answerable, out.citations = _from_submit(submit.arguments)
                break
            if not result.tool_calls:  # a plain-text reply is accepted too
                out.text, out.answerable, out.citations = parse_answer(result.text)
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": result.text or None,
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in result.tool_calls
                    ],
                }
            )
            for call in result.tool_calls:
                output = self._run_tool(call.name, call.arguments, out.retrieved)
                evidence.append(output)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        else:
            # A hard stop keeps a confused model from looping and running up cost.
            out.text, out.answerable = "No answer within the step limit.", False
            out.error = "max_steps"
        out.evidence = "\n\n".join(evidence)
        out.latency_s = time.perf_counter() - start
        return out


class DifySystem:
    """Calls a published Dify chat app. Give the app the same JSON answer format
    (see docs/dify_prototype.md) so it is graded the same way as the others."""

    def __init__(self, base_url: str, api_key: str, timeout: float = 120.0):
        import httpx

        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def answer(self, question: str) -> Answer:
        start = time.perf_counter()
        response = self._client.post(
            "/chat-messages",
            json={"inputs": {}, "query": question, "response_mode": "blocking", "user": "policyqa-eval"},
        )
        response.raise_for_status()
        data = response.json()
        text, answerable, citations = parse_answer(data.get("answer", ""))
        meta = data.get("metadata") or {}
        usage = meta.get("usage") or {}
        resources = meta.get("retriever_resources") or []
        return Answer(
            text=text,
            answerable=answerable,
            citations=citations,
            evidence="\n\n".join(str(r.get("content", "")) for r in resources),
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=time.perf_counter() - start,
            steps=1,
        )

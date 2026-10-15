# Code tour

A guided read of the project: what happened when it was run, one question followed through the code, then each file in reading order. About 30 minutes.

## 1. What was run

| Step | Result |
|---|---|
| Install (`pip install -e ".[dev]"`) | Clean |
| Tests (`pytest`) | 54 passed |
| Baseline (`python -m policyqa.evaluate --system extractive --retriever bm25`) | 15 of 40 correct (38%); the right section was in the top 5 for 98% of questions |
| Service (`uvicorn policyqa.api:create_app --factory`) | `/ask` and `/quote` both respond |

Two calls to the running service:

```
POST /quote  {"age": 35, "risk_class": "Preferred", "term_years": 20, "face_amount": 500000, "mode": "monthly"}
→ {"rate_per_1000": "1.01", "policy_fee": "60.00", "annual_premium": "565.00", "payment": "49.44"}

POST /quote  {"age": 52, "risk_class": "Standard", "term_years": 30, "face_amount": 500000}
→ 422 "the 30-year term is issued at ages 18 through 50; age 52 is not eligible"
```

The baseline uses no language model. It finds the best section by keyword search and returns the one sentence that shares the most words with the question.

## 2. One question, start to finish

Question L03: *"What is the grace period for a missed term life premium?"* The answer key says the answer must contain `31 day*`.

**Chunking** (`chunking.py`). The six documents become 40 chunks, one per section. Each chunk starts with its document and section title:

```
Larkspur Term Life: Policy Summary | Grace Period
A term premium not paid by its due date may still be paid during a grace period of 31 days. ...
```

**Tokenizing** (`retrieval.py`). The question is lowercased, common words are dropped, and plurals are trimmed:

```
['grace', 'period', 'missed', 'term', 'life', 'premium']
```

**Scoring** (`retrieval.py`, `BM25Retriever`). Each word has a weight that is higher the rarer it is across the 40 chunks:

```
missed 3.31   grace 2.46   period 1.36   term 1.27   premium 0.69   life 0.60
```

Every chunk gets the sum of the weights of the question words it contains, adjusted for how often each appears and how long the chunk is. The top five:

```
10.44  term_life#grace-period
 8.59  whole_life#grace-period
 8.55  billing#late-payment-notices
 4.88  term_life#free-look-period
 4.42  billing#refunds-on-cancellation
```

The term and whole life grace sections are close. The word "term" in the title line is part of what puts the right one first, which is why the title is added to every chunk.

**Answering** (`systems.py`, `ExtractiveSystem`). It takes the top chunk and returns the sentence with the most question words:

```
A term premium not paid by its due date may still be paid during a grace period of 31 days.
```

**Scoring the answer** (`metrics.py`). The answer contains "31 days", which matches `31 day*`, so it is correct. The gold section was retrieved first (reciprocal rank 1.0), it was cited, and the number 31 appears in the retrieved text (grounded).

## 3. The files, in reading order

### `chunking.py` (70 lines)

Splits each document at its `## ` headings. A section longer than 220 words is cut into overlapping windows; none of these sections is that long, so there is one chunk per section.

```python
@property
def text(self) -> str:
    """What gets indexed and shown to the model."""
    return f"{self.doc_title} | {self.heading}\n{self.body}"
```

Likely question: *why split on headings and not every N words?* A policy section is one rule. Cutting inside it can separate a rule from its exception.

### `retrieval.py` (173 lines)

Four retrievers with the same `search(query, k)` method, so the evaluation can swap them.

- `BM25Retriever`: keyword scoring, shown above. `k1` limits how much a repeated word keeps adding; `b` sets how strongly long chunks are penalized.
- `DenseRetriever`: turns every chunk and the question into a vector with a Hugging Face embedding model and ranks by cosine similarity. It matches meaning, so it can find "think it over" for "free look".
- `HybridRetriever`: runs both and combines them by rank:

```python
fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (self.rrf_k + rank)
```

- `RerankRetriever`: fetches 20 candidates, then a second Hugging Face model reads the question and each candidate together and reorders them. Slower and more accurate than embeddings, so it is only used on the short list.

Likely question: *why combine ranks?* BM25 scores (around 10 here) and cosine similarities (between -1 and 1) are on different scales, so adding them directly lets one drown out the other.

### `tools.py` (132 lines)

The premium calculator. Rates live in a SQL table; the lookup is one query:

```python
"SELECT rate_per_1000 FROM rates "
"WHERE ? BETWEEN age_min AND age_max AND risk_class = ? AND term_years = ?"
```

Then: annual premium = face amount / 1,000 × rate + $60 fee, and each payment = annual premium × a factor for the payment frequency. Every invalid request raises a `QuoteError` with a message written to be shown to the model.

The bottom of the file holds `QUOTE_TOOL` and `SEARCH_TOOL`: the descriptions the language model reads to decide when and how to call each tool.

Likely question: *why not let the model do the arithmetic?* Models make arithmetic slips and cannot see the rate table. A wrong premium is the costliest error this system could make.

### `llm.py` (113 lines)

One wrapper so every model is called the same way. `OpenAICompatModel.chat` sends messages, returns the text, any tool calls, token counts and timing. `ScriptedModel` replays canned replies so the agent can be tested without a real model.

### `systems.py` (252 lines)

The most important file. Three designs with the same `answer(question)` method:

| Class | Model calls | What it does |
|---|---|---|
| `ExtractiveSystem` | 0 | Returns the best sentence from the top chunk |
| `RagSystem` | 1 | Retrieves five chunks, puts them in the prompt, asks for an answer |
| `AgentSystem` | Up to 6 | The model chooses between searching, pricing, and answering |

Read the two prompts at the top (`RAG_SYSTEM_PROMPT`, `AGENT_SYSTEM_PROMPT`). Both demand a JSON reply with `answerable`, `answer` and `citations`, and both tell the model to say so when the material does not contain the answer.

The agent loop:

```python
for _ in range(self.max_steps):
    result = self.model.chat(messages, tools=[SEARCH_TOOL, QUOTE_TOOL])
    ...
    if not result.tool_calls:
        out.text, out.answerable, out.citations = parse_answer(result.text)
        break
    ...  # run each tool, append its output to the conversation, go round again
else:
    out.text, out.answerable = "No answer within the step limit.", False
```

Each round the model either asks for a tool or gives a final answer. Tool output, including error messages, goes back into the conversation for the next round.

Likely question: *what stops the agent looping forever?* `max_steps`. After six rounds it gives up and declines.

### `metrics.py` (100 lines)

Marks an answer right or wrong with fixed rules, no model involved.

- Answerable question: right if the system did not decline and the answer contains every required fact.
- Unanswerable question: right only if the system declined.
- `has_phrase` matches whole tokens, so `6%` does not match `36%`.
- `numeric_grounding` checks that every number in the answer appears in the retrieved text or tool output.

Likely question: *what is the weakness?* A correct answer in unexpected wording can be marked wrong. So the failures have to be read, not just counted.

### `stats.py` (61 lines)

- `bootstrap_ci`: resamples the 40 results 5,000 times to get a range for accuracy. The baseline's 38% comes with a range of 22% to 52%.
- `paired_bootstrap_diff` and `mcnemar_exact`: test whether one system really beats another on the same questions.

Likely question: *is a 5-point gap between two models meaningful here?* No. One question is 2.5 points, and the interval is about 15 points either side.

### `evaluate.py` (83 lines) and `report.py` (121 lines)

`evaluate.py` builds one system, runs all questions, scores each, and saves a line per question to `results/`. `report.py` reads every saved run and writes the comparison table.

### `api.py` (79 lines), `mcp_server.py` (51 lines), `factory.py` (53 lines)

`api.py` wraps a system in a web service. `mcp_server.py` offers the same two tools to any MCP client. `factory.py` builds a system from settings and is shared by all three entry points.

## 4. Why the baseline's 25 wrong answers were wrong

| Cause | Count | Questions |
|---|---|---|
| Right section on top, wrong sentence picked | 6 | L06, L09, L10, L12, L13, L15 |
| Wrong section ranked first | 1 | L05 |
| Needs two sections | 6 | M01 to M06 |
| Needs the rate table | 6 | T01 to T06 |
| Should have declined | 6 | U01 to U06 |

Three worth knowing:

- **L09**, "surrender charge in policy year 3". It returned "A surrender charge applies if a whole life policy is surrendered in its first five years." The right section, but the 6% is in the next sentence. A model reading the whole section should get this.
- **L05**, "deadline for converting a term policy to whole life". The term life overview outranked the conversion section, which came second. The question's words ("term", "policy", "whole", "life") are common, and the section says "conversion", not "converting". This is the kind of miss embeddings are meant to fix.
- **U04**, "how much does the Waiver of Premium Rider cost". It answered "One premium of $72 per year covers all eligible children." That is the price of a different rider, stated confidently. It is the most dangerous failure in the set and the one to watch in every model run.

The pattern: 12 of the 25 failures (the first and last rows) are reading problems, not search problems. That is the gap a language model is there to close.

## 5. Check yourself

- **What do BM25's `k1` and `b` control?** `k1`: how much repeated uses of a word keep adding to the score. `b`: how much long chunks are penalized.
- **Why does hybrid search combine ranks?** The two scores are on different scales.
- **Why 68% on lookups and 0% elsewhere?** One sentence from one section cannot combine two sections, cannot see the rate table, and never declines.

## 6. Still to do yourself

- Create a Hugging Face account and a token with permission to call Inference Providers.
- Create a GitHub repository and push the project.
- Run it once on your own machine. The first time an interviewer asks you to show it, it should not be the first time you have started it.

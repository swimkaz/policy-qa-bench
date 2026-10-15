# Policy QA Bench

A question-answering system for life insurance policy documents, built to answer one question with evidence: **which combination of retriever, system design and model is good enough to ship, and how do we know?**

The documents describe Larkspur Mutual Life, a fictional insurer. All policy text, rates and questions are synthetic.

## What is in it

| Part | What it does |
|---|---|
| Retrieval | BM25, dense embeddings from Hugging Face, hybrid (reciprocal rank fusion), and a cross-encoder reranker |
| Systems | An extractive floor with no model, single-shot RAG, a tool-calling agent, and a Dify app called over its API |
| Tools | Document search and a premium quote backed by SQL and exact decimal arithmetic, also exposed over MCP |
| Evaluation | 40 questions with an answer key, deterministic scoring, bootstrap confidence intervals, paired significance tests |
| Service | FastAPI app with a Dockerfile |

## The test set

| Type | Count | What it tests |
|---|---|---|
| Lookup | 22 | One fact from one section. Several facts differ by product (free look is 20 days on term, 10 on whole life), so the wrong section gives a wrong answer |
| Multi | 6 | Needs two sections, sometimes plus a small inference |
| Tool | 6 | Premium amounts. The rate table is not in the documents, so only a system that calls the quote tool can be right |
| Unanswerable | 6 | The documents do not contain the answer. The only correct response is to say so |

`tests/test_golden.py` checks the answer key against the documents and the rate table, so a wrong key fails the build.

## Results

40 questions, hybrid retrieval, models served through Hugging Face Inference Providers. Full table with confidence intervals in `results/REPORT.md`; every wrong answer is explained in `ERROR_ANALYSIS.md`; conclusions in `FINDINGS.md`.

| System | gpt-oss-120b | gpt-oss-20b | Llama 3.3 70B |
|---|---|---|---|
| Baseline, no LLM | 38% | 38% | 38% |
| Single-shot RAG | 85% | 85% | 82% |
| Tool-calling agent | 92% | 88% | 95% |

- **Retrieval:** hybrid search had the best MRR (0.96 against 0.91 for BM25); every method reached 98% recall@5. The cross-encoder reranker was ten times slower and slightly worse.
- **Agent against RAG:** the agent's gain is entirely on premium questions, which need the pricing tool. RAG declined all of them instead of guessing.
- **Stopping:** as an agent, gpt-oss-20b kept searching until the step limit on 5 of 6 unanswerable questions. As RAG, it declined all 6.
- **No false facts:** across 240 LLM answers, every wrong answer was a refusal, an unfinished search, or a formatting failure.
- **Significance:** no difference against the RAG baseline is statistically significant (McNemar p > 0.18).

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                                    # 60 tests, no network needed
python -m policyqa.evaluate --system extractive --retriever bm25
```

Hugging Face retrievers and models:

```bash
pip install -e ".[hf]"                                    # sentence-transformers
export HF_TOKEN=...                                       # needs Inference Providers permission
python -m policyqa.evaluate --system extractive --retriever hybrid-rerank
python -m policyqa.evaluate --system rag   --retriever hybrid --model gpt-oss-120b
python -m policyqa.evaluate --system agent --retriever hybrid --model gpt-oss-120b
python -m policyqa.report --baseline rag_hybrid_k5_gpt-oss-120b
python -m policyqa.rescore                                # re-grade saved results, no model calls
```

Models are listed in `models.toml`. Any OpenAI-compatible endpoint works.

Service and MCP server:

```bash
uvicorn policyqa.api:create_app --factory --port 8000     # POST /ask, POST /quote, GET /docs
python -m policyqa.mcp_server                             # stdio
docker build -t policyqa . && docker run -p 8000:8000 policyqa
```

## Metrics

| Column | Meaning |
|---|---|
| Accuracy | Answerable: the answer contains every required fact. Unanswerable: the system declined |
| Recall | Share of the gold sections that were retrieved |
| MRR | Mean of 1 / rank of the first gold section |
| Cited | Share of answers that cite a gold section |
| Grounded | Share of answers whose every number appears in the retrieved text or tool output |
| False abstain | Share of answerable questions the system declined |

## Design decisions

- **Chunk on section headings.** A policy section is one rule. Splitting inside it separates a rule from its exception.
- **Put the document and section title in every chunk.** "31 days" means nothing until the chunk says which product it is about.
- **Fuse ranks, not scores.** BM25 and cosine scores are on different scales; reciprocal rank fusion needs no tuning to combine them.
- **Prices come from a tool.** Models make arithmetic errors and a wrong premium is costly, so pricing is a SQL lookup with decimal rounding.
- **Deterministic scoring first.** Every score is reproducible and every failure can be read. The cost is rigidity, covered below.
- **Report intervals.** With 40 questions, one question is 2.5 points. The report gives a confidence interval on accuracy and a paired test on differences.
- **An agent step limit.** A hard cap stops a confused model from looping and running up cost.

## Limitations

- The corpus is small and synthetic. Real policy documents are longer, messier and inconsistent.
- 40 questions give wide intervals. Small differences between models will not be distinguishable.
- String-rule scoring can reject a correct answer phrased unexpectedly. Read the failures in `results/*.jsonl` before trusting a number.
- The grounding check flags numbers the model worked out itself, even when the arithmetic is right.
- Each run is a single pass. Models are not fully deterministic, so repeat runs to see run-to-run variance.

## What has been verified

Run end to end on Windows: the test suite, all four retrievers including the Hugging Face embedding and reranking models, all six model runs, and the API. The MCP server was tested with a real MCP client. The Docker image and the Dify integration have not been run.

## Layout

```
data/corpus/        six policy documents
data/rates.csv      term life rate table
data/golden.jsonl   questions and answer key
src/policyqa/       chunking, retrieval, tools, llm, systems, metrics, stats, evaluate, report, rescore, api, mcp_server
tests/              60 tests
docs/               Dify prototype guide
PLAN.md             six-day plan and interview notes
FINDINGS.md         one-page conclusions
ERROR_ANALYSIS.md   every wrong answer, by cause
```

# Findings

*Draft written from the benchmark results. Rewrite it in your own words before sharing it.*

## Recommendation

Answer document questions with single-shot RAG and route premium questions to the pricing tool, not to a free-running agent. If an agent is used, it needs an explicit rule for when to stop searching. A small model (gpt-oss-20b) is accurate enough for document questions, and no model in the test produced a false fact.

## What was tested

40 questions about six life insurance policy documents, with an answer key: 22 single-fact lookups, 6 needing two sections, 6 premium calculations, and 6 that can't be answered from the documents. Three open-weight models, each tried two ways: single-shot RAG, and an agent that can search repeatedly and call a pricing tool.

## Three numbers

| | Single-shot RAG | Agent |
|---|---|---|
| Overall accuracy | 82–85% | 88–95% |
| Premium questions | 0%; correctly declined every one | 100% |
| Questions with no answer in the documents | 100% declined | 17–83% |

## What it means

1. **The agent's advantage is pricing, and only pricing.** It answered all premium questions by calling the pricing tool. On the other 34 questions, RAG was as good or better.
2. **Agents can fail to stop.** Given the option to keep searching, gpt-oss-20b looked for answers that didn't exist until it hit the step limit on 5 of 6 unanswerable questions. The same model declined all 6 correctly as single-shot RAG.
3. **No hallucinated facts.** Across 240 LLM answers, every wrong answer was a refusal, an unfinished search, a formatting problem, or an incomplete answer. None stated something false. All six runs avoided the trap question, where a different rider's price sat next to the one asked about.
4. **Model size didn't matter much.** gpt-oss-20b matched gpt-oss-120b as RAG. None of the differences measured against the RAG baseline was statistically significant (McNemar p > 0.18), so the test can't pick a winner.
5. **Search was not the bottleneck.** Hybrid search put the right section in the top five 98% of the time. A reranker made it slower and slightly worse.

## Risks before production

- **Small, synthetic test set.** 40 questions give confidence intervals of about ±10 points. A real deployment needs hundreds of real customer questions.
- **Format failures.** gpt-oss-120b occasionally called a tool that doesn't exist (2 of 40 questions). This needs a retry or a fallback.
- **Step limits.** Agent runs that hit the limit cost six model calls and up to 5 seconds before failing.
- **No access control or human review yet.** Anything acting on a customer's policy would need both.

## Next steps

1. Add a stopping rule to the agent ("after two searches with no answer, decline") and re-measure.
2. Retry invalid tool calls once with a reminder of the valid tools.
3. Expand the test set with paraphrased and real questions.

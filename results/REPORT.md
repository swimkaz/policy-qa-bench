# Benchmark report

| Run | Accuracy (95% CI) | Lookup | Multi | Tool | Unanswerable | Recall | MRR | Cited | Grounded | False abstain | p50 s | p95 s | Tokens/q | Cost $ | Errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| agent_hybrid_k5_gpt-oss-120b | 92% (82% to 100%) | 95% | 100% | 100% | 67% | 96% | 0.90 | 96% | 93% | 0% | 1.49 | 3.83 | 2360 | 0.0000 | 3 |
| agent_hybrid_k5_gpt-oss-20b | 88% (78% to 98%) | 100% | 100% | 100% | 17% | 96% | 0.94 | 100% | 94% | 0% | 1.08 | 5.00 | 2777 | 0.0000 | 5 |
| agent_hybrid_k5_llama-3.3-70b | 95% (88% to 100%) | 100% | 83% | 100% | 83% | 98% | 0.94 | 68% | 97% | 0% | 1.88 | 2.53 | 2263 | 0.0000 | 0 |
| extractive_bm25_k5 | 38% (22% to 52%) | 68% | 0% | 0% | 0% | 98% | 0.91 | 86% | 100% | 0% | 0.00 | 0.00 | 0 | 0.0000 | 0 |
| extractive_dense_k5 | 38% (22% to 52%) | 68% | 0% | 0% | 0% | 98% | 0.95 | 93% | 100% | 0% | 0.03 | 0.04 | 0 | 0.0000 | 0 |
| extractive_hybrid-rerank_k5 | 38% (22% to 52%) | 68% | 0% | 0% | 0% | 98% | 0.93 | 89% | 100% | 0% | 0.31 | 0.35 | 0 | 0.0000 | 0 |
| extractive_hybrid_k5 | 38% (22% to 52%) | 68% | 0% | 0% | 0% | 98% | 0.96 | 93% | 100% | 0% | 0.03 | 0.04 | 0 | 0.0000 | 0 |
| rag_hybrid_k5_gpt-oss-120b | 85% (72% to 95%) | 100% | 100% | 0% | 100% | 98% | 0.96 | 100% | 96% | 18% | 0.87 | 1.23 | 805 | 0.0000 | 0 |
| rag_hybrid_k5_gpt-oss-20b | 85% (72% to 95%) | 100% | 100% | 0% | 100% | 98% | 0.96 | 100% | 96% | 18% | 0.65 | 1.12 | 842 | 0.0000 | 0 |
| rag_hybrid_k5_llama-3.3-70b | 82% (70% to 92%) | 100% | 83% | 0% | 100% | 98% | 0.96 | 100% | 96% | 21% | 0.96 | 1.42 | 654 | 0.0000 | 0 |

## Paired comparison

| Run vs rag_hybrid_k5_gpt-oss-120b | Accuracy difference (95% CI) | Wins | Losses | McNemar p |
|---|---|---|---|---|
| agent_hybrid_k5_gpt-oss-120b | +8 pts (-8 to +22) | 6 | 3 | 0.508 |
| agent_hybrid_k5_gpt-oss-20b | +2 pts (-12 to +20) | 6 | 5 | 1.000 |
| agent_hybrid_k5_llama-3.3-70b | +10 pts (-2 to +22) | 6 | 2 | 0.289 |
| extractive_bm25_k5 | -48 pts (-62 to -32) | 0 | 19 | 0.000 |
| extractive_dense_k5 | -48 pts (-62 to -32) | 0 | 19 | 0.000 |
| extractive_hybrid-rerank_k5 | -48 pts (-62 to -32) | 0 | 19 | 0.000 |
| extractive_hybrid_k5 | -48 pts (-62 to -32) | 0 | 19 | 0.000 |
| rag_hybrid_k5_gpt-oss-20b | +0 pts (+0 to +0) | 0 | 0 | 1.000 |
| rag_hybrid_k5_llama-3.3-70b | -2 pts (-8 to +0) | 0 | 1 | 1.000 |

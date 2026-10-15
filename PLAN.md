# Six-day plan

The code is scaffolded and tested. Your job this week is to run the experiments, form your own conclusions, and be able to explain every file. An interviewer will find out quickly whether the conclusions are yours.

If asked how you built it, say so plainly: the documents are synthetic, you used an AI assistant to scaffold the code, and you ran the experiments and did the analysis. That is a normal way to work, and it is better than being caught on a detail.

## Day 1: run it and read it

1. Follow the quick start in `README.md`. Get `pytest` passing and the baseline run producing 38%.
2. Create a Hugging Face account and a token with Inference Providers permission. Create a GitHub repository and push the code.
3. Read the source in this order, about 900 lines in total:
   `chunking.py`, `retrieval.py`, `tools.py`, `systems.py`, `metrics.py`, `stats.py`, `evaluate.py`.
4. Open `results/extractive_bm25_k5.jsonl` and read the wrong answers.

You are done when you can explain, without notes: what BM25's `k1` and `b` control, why reciprocal rank fusion uses ranks, and why the baseline gets 68% on lookups but 0% on everything else.

## Day 2: retrieval with Hugging Face models

```bash
pip install -e ".[hf]"
for r in bm25 dense hybrid hybrid-rerank; do
  python -m policyqa.evaluate --system extractive --retriever $r
done
python -m policyqa.report --baseline extractive_bm25_k5
```

The extractive system costs nothing, so this compares retrievers for free. Look at Recall and MRR, not accuracy.

BM25 already reaches 98% recall, because the questions reuse the documents' wording. So dense retrieval will probably show no gain. That is a real finding, and the next step makes it more interesting:

- **Write 10 to 15 paraphrased questions yourself** that avoid the documents' vocabulary ("How long can I think it over before the policy is final?" for the free look period). Add them to `data/golden.jsonl` with the same fields. `pytest` will check your answer key.
- Rerun. Paraphrases are where embeddings should beat keyword matching.
- Try a second embedding model with `--embedder <model id>` and compare.

Writing the questions yourself matters. "How did you build the test set?" is the question this project most invites.

## Day 3: the model benchmark

Check `models.toml`: confirm each model is still served and supports tool calling, add a third model, and fill in prices.

```bash
python -m policyqa.evaluate --system rag --retriever hybrid --model gpt-oss-120b --limit 5   # plumbing check
for m in gpt-oss-120b gpt-oss-20b; do
  python -m policyqa.evaluate --system rag   --retriever hybrid --model $m
  python -m policyqa.evaluate --system agent --retriever hybrid --model $m
done
python -m policyqa.report --baseline rag_hybrid_k5_gpt-oss-120b
```

Then spend more time reading than running:

- Read every wrong answer. Sort each into: retrieval missed, model misread the right section, model answered when it should have declined, or the scoring rule rejected a correct answer.
- If the rule was wrong, fix the rule in `golden.jsonl` and note the change. Do this before comparing models, and apply it to all of them.
- Rerun one configuration three times to see how much accuracy moves between identical runs.

Questions to answer in your own words:

- Does the agent beat single-shot RAG outside the tool questions, or only on them?
- Does the smaller model lose accuracy, and is the gap bigger than the confidence interval?
- What does each correct answer cost, and how long does it take?
- Which failure would worry an insurer most? (A confident wrong number, most likely. Check the Unanswerable column and U04 in particular.)

## Day 4: the Dify prototype

Follow `docs/dify_prototype.md`. Budget half a day. Score it with the same harness and add it to the report.

## Day 5: publish and write up

- Push everything to GitHub with the final `results/REPORT.md`.
- Publish the question set as a Hugging Face dataset with a dataset card that covers how it was built, what it tests and its limits. Check the current upload instructions at https://huggingface.co/docs/hub/datasets-adding.
- Write `FINDINGS.md`: one page for a non-technical reader. Recommendation first, then three numbers, then risks. The posting asks for this skill by name.
- If time allows: port `AgentSystem` to AWS Strands. The posting lists it, the tools are already separate functions, and the harness will tell you whether the port behaves the same.

## Day 6: rehearse

Prepare a two-minute walkthrough: the problem, the test set, the three systems, the headline result, what you would do next. Then practise the questions below out loud.

## Questions to expect

| Question | Where the answer is |
|---|---|
| How did you define success before building? | The four question types and the metrics table in the README. Unanswerable questions exist because a wrong answer costs more than no answer |
| How do you know the answer key is right? | `tests/test_golden.py` checks it against the documents and the rate table |
| Why not use a model as the grader? | Deterministic rules are reproducible and free. The weakness is rigidity. A model grader is a reasonable next step once it has been checked against human labels on a sample |
| Is the difference between two models real? | Paired bootstrap interval and McNemar's test in `stats.py`. With 40 questions, small gaps are not distinguishable, and you say so |
| Why hybrid retrieval? | Keyword search catches exact terms and numbers; embeddings catch paraphrases. Your day 2 numbers show when each matters |
| Why an agent and not just RAG? | The rate table is not in the documents. Your day 3 numbers show whether the agent helps elsewhere and what it costs in latency and tokens |
| How do you stop it making up a premium? | The quote tool, the instruction never to estimate, tool errors passed back to the model, and the grounding metric |
| What would production need? | Access control, logging and tracing, monitoring answer quality over time, a human review path for low-confidence answers, a larger test set built from real questions, load testing |
| What about responsible AI? | Abstention is scored. Every answer cites its source. Numbers are traced to evidence. Decisions about a customer's coverage stay with a person |
| What went wrong? | Have a real story from days 2 to 4. A specific failure you diagnosed is the most convincing thing you can bring |

## What this project does not cover

The posting also asks for breadth in classical machine learning, causal inference, optimisation and Bayesian methods, and for cloud deployment. This project touches statistics and deployment lightly and the others not at all. Prepare examples from your past work for those.

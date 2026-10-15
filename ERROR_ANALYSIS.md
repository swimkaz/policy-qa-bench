# Error analysis

Every wrong answer from the six LLM runs (3 models × RAG and agent, 40 questions each, 240 answers), sorted by cause. Scores are after two grader fixes described at the end. Review each row yourself before relying on it.

## Summary

| Cause | Count | Runs affected |
|---|---|---|
| Declined a pricing question because it has no pricing tool | 18 | All three RAG runs, by design |
| Agent kept searching and hit the 6-step limit without answering | 6 | gpt-oss-20b agent (5), gpt-oss-120b agent (1) |
| Model called a tool named `json` that doesn't exist; the provider rejected the call | 2 | gpt-oss-120b agent |
| Search found only one of the two sections needed, so the model declined | 1 | Llama RAG |
| Answer incomplete: gave the rule but not the calculated limit | 1 | Llama agent |
| Answer text declined correctly, but the `answerable` flag said true | 1 | Llama agent |
| **Confident wrong answer (a false fact)** | **0** | None |

**Question U04**, the trap where the Child Term Rider's $72 sits next to the Waiver of Premium Rider: all six runs correctly said the cost isn't stated.

## Per run

### Agent, gpt-oss-120b: 92% (3 wrong)

| Question | What happened | Cause |
|---|---|---|
| L12 | Tried to submit its answer through a tool named `json`; the provider returned a 400 error. The answer it tried to give was correct | Invalid tool call |
| U03 | Same `json` tool error. The refusal it tried to give was correct | Invalid tool call |
| U05 | Searched 6 times for a claims phone number that doesn't exist and never submitted | Step limit |

### Agent, gpt-oss-20b: 88% (5 wrong)

| Question | What happened | Cause |
|---|---|---|
| U01 to U05 | On five of the six unanswerable questions, it kept searching until the 6-step limit and never submitted an answer | Step limit |

The same model scored 100% on unanswerable questions as single-shot RAG, where it has no option to keep searching.

### Agent, Llama 3.3 70B: 95% (2 wrong)

| Question | What happened | Cause |
|---|---|---|
| M05 | Said the limit is 20 times income but didn't work out $1,600,000 or say $2,000,000 exceeds it | Incomplete |
| U06 | Wrote "the tool cannot calculate the premium for a 62-year-old", a correct refusal, but set `answerable: true` | Wrong flag |

### RAG, gpt-oss-120b and gpt-oss-20b: 85% each (6 wrong each)

| Question | What happened | Cause |
|---|---|---|
| T01 to T06 | Said the documents don't contain premium rates, which is true | No pricing tool, by design |

### RAG, Llama 3.3 70B: 82% (7 wrong)

| Question | What happened | Cause |
|---|---|---|
| T01 to T06 | Declined, as above | No pricing tool, by design |
| M04 | Search returned the underwriting section but not the term-length section, so it declined | Search miss, safe decline |

## Grader problems found and fixed

1. **Errors counted as correct refusals.** A failed call on an unanswerable question was scored as a correct "I don't know". Fixed: an error is always wrong. This lowered the gpt-oss-20b agent from 98% to 85%.
2. **Typographic hyphens.** gpt-oss writes "paid‑up additions" with a non-breaking hyphen (U+2011), which the matching rule didn't recognise, so four correct answers to L10 were marked wrong. Fixed by normalising typographic characters. This raised four runs by 2.5 points each.

Both fixes have tests, and `python -m policyqa.rescore` re-grades saved results without calling any model.

## Open judgement calls

- **Step-limit refusals.** When the agent hits the step limit on an unanswerable question, it ends up declining, which is the right outcome reached the wrong way. They're counted as wrong here, because the agent didn't decide anything and spent six model calls doing it. Counting them as correct would put the gpt-oss-20b agent at 100% on unanswerable questions.
- **The Llama U06 flag.** The text is a correct refusal but the structured flag is wrong. It's counted as wrong because any system reading the flag would treat it as an answer.

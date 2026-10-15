# The Dify prototype

The goal is a working version of the same question-answering flow in Dify, scored on the same 40 questions as the coded version. The story for an interview: "I prototyped in an afternoon, measured it, and the measurement told me what to build properly."

This guide has not been run against a live Dify workspace, and Dify's menus change between releases. Treat the names below as a description of what to configure and check the current docs at https://docs.dify.ai if a label differs.

## 1. Knowledge base

1. Create a knowledge base and upload the six files in `data/corpus/`.
2. Chunking: split on section headings. Set the delimiter to `\n## ` and a maximum chunk length large enough to hold a whole section (about 1,000 tokens).
3. Indexing: choose the high-quality (embedding) index and pick an embedding model.
4. Retrieval: hybrid search with reranking, top 5.
5. Use the retrieval test screen with five questions from `data/golden.jsonl`. Check that L01 and L02 (free look on term versus whole life) return different sections.

Write down every setting. "Why that chunk size?" is a likely question.

## 2. Chatflow, version 1: retrieval only

Nodes: Start, Knowledge Retrieval, LLM, Answer.

In the LLM node, use the system prompt from `RAG_SYSTEM_PROMPT` in `src/policyqa/systems.py`, including the JSON answer format. The scoring code reads that JSON, so keeping the format is what makes the two versions comparable. Pass the retrieval result into the prompt as context.

## 3. Chatflow, version 2: add the quote tool

The quote tool is the `/quote` endpoint of this project's API.

1. Start the API: `uvicorn policyqa.api:create_app --factory --port 8000`.
2. Make it reachable from Dify. Self-hosted Dify on the same machine can call it directly. Dify Cloud needs a public URL, so deploy the container or use a tunnel.
3. In Dify, add a custom tool by importing the OpenAPI schema from `/openapi.json`. Keep only the `quote_term_premium` operation.
4. Replace the LLM node with an Agent node that has the knowledge base and the quote tool, or add a question classifier that routes premium questions to an HTTP Request node. Try both; which is more reliable is a finding.

## 4. Score it

Publish the app, create an API key, then:

```bash
export DIFY_BASE_URL=https://api.dify.ai/v1      # or your self-hosted URL ending in /v1
export DIFY_API_KEY=app-...
python -m policyqa.evaluate --system dify --limit 5   # check the plumbing first
python -m policyqa.evaluate --system dify
python -m policyqa.report --baseline dify
```

What is and is not comparable:

- **Accuracy, false abstentions, latency** are comparable.
- **Recall, MRR and Cited** are not. Dify's chunks do not carry this project's section ids, so those columns read n/a or 0 for the Dify run.

## 5. What to take from it

Have an answer to each of these:

- Where did the prototype score lower than the code, and why?
- What could you not control in Dify that you could in code? (Candidates: exact chunk boundaries, the fusion method, the step limit, error handling for tool failures, tests.)
- What did Dify give you faster? (Candidates: a usable chat interface, document ingestion, a shareable demo for stakeholders.)
- When would you leave it in Dify and not rewrite it?

Export the app as YAML and commit it to `docs/` so the prototype is versioned with the code.

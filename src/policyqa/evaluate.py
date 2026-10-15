"""Run one system over the golden questions and save one scored record per question.

    python -m policyqa.evaluate --system extractive --retriever bm25
    python -m policyqa.evaluate --system rag   --retriever hybrid --model gpt-oss-120b
    python -m policyqa.evaluate --system agent --retriever hybrid --model gpt-oss-120b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import data_dir
from .factory import SYSTEMS, build_system
from .metrics import score
from .report import format_summary, summarize
from .retrieval import DEFAULT_EMBEDDER, DEFAULT_RERANKER, RETRIEVERS
from .systems import Answer, QASystem


def load_golden(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(system: QASystem, items: list[dict], price_in: float = 0.0, price_out: float = 0.0, has_retrieval: bool = True) -> list[dict]:
    records = []
    for item in items:
        try:
            answer = system.answer(item["question"])
        except Exception as exc:  # one failed call must not lose the whole run
            answer = Answer(text="", answerable=bool(item["answerable"]), error=f"{type(exc).__name__}: {exc}")
        records.append(score(item, answer, price_in, price_out, has_retrieval))
        print("." if records[-1]["correct"] else "x", end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", choices=SYSTEMS, default="extractive")
    parser.add_argument("--retriever", choices=RETRIEVERS, default="bm25")
    parser.add_argument("--model", help="a name from models.toml (rag and agent only)")
    parser.add_argument("--embedder", default=DEFAULT_EMBEDDER, help="Hugging Face embedding model id")
    parser.add_argument("--reranker", default=DEFAULT_RERANKER, help="Hugging Face cross-encoder model id")
    parser.add_argument("--k", type=int, default=5, help="sections retrieved per search")
    parser.add_argument("--golden", type=Path, default=data_dir() / "golden.jsonl")
    parser.add_argument("--out", type=Path, default=data_dir().parent / "results")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    args = parser.parse_args(argv)

    system, model = build_system(args.system, args.retriever, args.model, args.k, args.embedder, args.reranker)
    items = load_golden(args.golden)[: args.limit]
    parts = [args.system] + ([] if args.system == "dify" else [args.retriever, f"k{args.k}"])
    if args.embedder != DEFAULT_EMBEDDER and args.retriever != "bm25":
        parts.append(args.embedder.split("/")[-1])  # keep runs with different embedders apart
    run_id = "_".join(parts + ([model.name] if model else []))

    records = run(
        system,
        items,
        price_in=model.price_in if model else 0.0,
        price_out=model.price_out if model else 0.0,
        has_retrieval=args.system != "dify",
    )
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{run_id}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({"run": run_id, **record}) + "\n")

    print(format_summary([summarize(run_id, records)]))
    errors = sum(1 for r in records if r["error"])
    if errors:
        print(f"\n{errors} question(s) ended in an error; see the 'error' field in {out_path}")
    print(f"\nSaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

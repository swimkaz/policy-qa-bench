"""Compare every saved run and write results/REPORT.md.

    python -m policyqa.report --baseline extractive_bm25_k5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data_dir
from .stats import bootstrap_ci, mcnemar_exact, mean, paired_bootstrap_diff, percentile

TYPES = ("lookup", "multi", "tool", "unanswerable")


def _rate(values: list) -> float | None:
    values = [float(v) for v in values if v is not None]
    return mean(values) if values else None


def is_correct(record: dict) -> bool:
    """Errors count as wrong. Re-checked here so result files saved by older
    versions, which scored an error on an unanswerable question as correct, are fixed too."""
    return bool(record["correct"]) and not record.get("error")


def summarize(run_id: str, records: list[dict]) -> dict:
    correct = [float(is_correct(r)) for r in records]
    lo, hi = bootstrap_ci(correct)
    answerable = [r for r in records if r["type"] != "unanswerable"]
    return {
        "run": run_id,
        "n": len(records),
        "accuracy": mean(correct),
        "ci": (lo, hi),
        "by_type": {t: _rate([is_correct(r) for r in records if r["type"] == t]) for t in TYPES},
        "errors": sum(1 for r in records if r.get("error")),
        "recall": _rate([r["recall"] for r in records]),
        "mrr": _rate([r["rr"] for r in records]),
        "cited": _rate([r["cited"] for r in records]),
        "grounded": _rate([r["grounded"] for r in records]),
        "false_abstain": _rate([r["false_abstain"] for r in answerable]),
        "p50_latency": percentile([r["latency_s"] for r in records], 50),
        "p95_latency": percentile([r["latency_s"] for r in records], 95),
        "tokens": mean([r["prompt_tokens"] + r["completion_tokens"] for r in records]),
        "cost": sum(r["cost_usd"] for r in records),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.0f}%"


def format_summary(summaries: list[dict]) -> str:
    header = (
        "| Run | Accuracy (95% CI) | Lookup | Multi | Tool | Unanswerable | Recall | MRR | "
        "Cited | Grounded | False abstain | p50 s | p95 s | Tokens/q | Cost $ | Errors |"
    )
    lines = [header, "|" + "---|" * 16]
    for s in summaries:
        lo, hi = s["ci"]
        mrr = "n/a" if s["mrr"] is None else f"{s['mrr']:.2f}"
        lines.append(
            f"| {s['run']} | {_pct(s['accuracy'])} ({_pct(lo)} to {_pct(hi)}) | "
            + " | ".join(_pct(s["by_type"][t]) for t in TYPES)
            + f" | {_pct(s['recall'])} | {mrr} | {_pct(s['cited'])} | {_pct(s['grounded'])} | "
            f"{_pct(s['false_abstain'])} | {s['p50_latency']:.2f} | {s['p95_latency']:.2f} | "
            f"{s['tokens']:.0f} | {s['cost']:.4f} | {s['errors']} |"
        )
    return "\n".join(lines)


def format_pairwise(runs: dict[str, list[dict]], baseline: str) -> str:
    base = {r["id"]: is_correct(r) for r in runs[baseline]}
    lines = [
        f"| Run vs {baseline} | Accuracy difference (95% CI) | Wins | Losses | McNemar p |",
        "|---|---|---|---|---|",
    ]
    for run_id, records in runs.items():
        if run_id == baseline:
            continue
        shared = [r for r in records if r["id"] in base]
        a = [is_correct(r) for r in shared]
        b = [bool(base[r["id"]]) for r in shared]
        diff, lo, hi = paired_bootstrap_diff([float(x) for x in a], [float(x) for x in b])
        wins = sum(1 for x, y in zip(a, b) if x and not y)
        losses = sum(1 for x, y in zip(a, b) if y and not x)
        lines.append(
            f"| {run_id} | {100 * diff:+.0f} pts ({100 * lo:+.0f} to {100 * hi:+.0f}) | "
            f"{wins} | {losses} | {mcnemar_exact(a, b):.3f} |"
        )
    return "\n".join(lines)


def load_runs(results_dir: Path) -> dict[str, list[dict]]:
    runs = {}
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            runs[path.stem] = [json.loads(line) for line in f if line.strip()]
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=data_dir().parent / "results")
    parser.add_argument("--baseline", help="run to compare the others against")
    args = parser.parse_args(argv)

    runs = load_runs(args.results)
    if not runs:
        print(f"no runs found in {args.results}")
        return 1
    sections = ["# Benchmark report", "", format_summary([summarize(k, v) for k, v in runs.items()])]
    if args.baseline:
        if args.baseline not in runs:
            print(f"baseline {args.baseline!r} not found; runs: {', '.join(runs)}")
            return 1
        sections += ["", "## Paired comparison", "", format_pairwise(runs, args.baseline)]
    text = "\n".join(sections) + "\n"
    (args.results / "REPORT.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

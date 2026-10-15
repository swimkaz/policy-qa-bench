"""Re-grade saved results against the current answer key, without calling any model.

Use it after fixing a grading rule or the answer key:

    python -m policyqa.rescore

Only the right/wrong verdicts are recomputed; retrieval, latency and token
figures are kept as they were recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data_dir
from .metrics import satisfies


def rescore_record(record: dict, item: dict) -> dict:
    abstained = bool(record["abstained"])
    if item["answerable"]:
        correct = (not abstained) and satisfies(record.get("answer", ""), item["must_include"])
    else:
        correct = abstained
    if record.get("error"):
        correct = False
    return {**record, "correct": bool(correct), "false_abstain": bool(item["answerable"] and abstained)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=data_dir().parent / "results")
    parser.add_argument("--golden", type=Path, default=data_dir() / "golden.jsonl")
    args = parser.parse_args(argv)

    with open(args.golden, encoding="utf-8") as f:
        items = {q["id"]: q for q in (json.loads(line) for line in f if line.strip())}
    for path in sorted(args.results.glob("*.jsonl")):
        with open(path, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        updated = [rescore_record(r, items[r["id"]]) if r["id"] in items else r for r in records]
        changed = [f'{r["id"]} {"wrong->right" if n["correct"] else "right->wrong"}'
                   for r, n in zip(records, updated) if r["correct"] != n["correct"]]
        with open(path, "w", encoding="utf-8") as f:
            for r in updated:
                f.write(json.dumps(r) + "\n")
        print(f"{path.stem}: {', '.join(changed) if changed else 'no change'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

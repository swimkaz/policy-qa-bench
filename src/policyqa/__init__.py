"""Policy question answering with a benchmark across retrievers, systems and models."""

import os
from pathlib import Path

__version__ = "0.1.0"


def data_dir() -> Path:
    """Folder holding corpus/, rates.csv and golden.jsonl (override with POLICYQA_DATA)."""
    env = os.environ.get("POLICYQA_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"

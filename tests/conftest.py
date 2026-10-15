import json

import pytest

from policyqa import data_dir
from policyqa.chunking import load_corpus
from policyqa.tools import RateBook


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(data_dir() / "corpus")


@pytest.fixture(scope="session")
def rates():
    return RateBook(data_dir() / "rates.csv")


@pytest.fixture(scope="session")
def golden():
    with open(data_dir() / "golden.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

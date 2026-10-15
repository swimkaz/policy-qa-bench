"""The premium quote tool: a SQL lookup plus exact decimal arithmetic.

Language models are unreliable at arithmetic and have no access to the rate
table, so pricing is a deterministic tool the agent calls. Money is computed
with Decimal because binary floats mis-round values such as 2.135.
"""

from __future__ import annotations

import csv
import sqlite3
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

RISK_CLASSES = ("Preferred Plus", "Preferred", "Standard", "Tobacco")
MODAL_FACTORS = {
    "annual": Decimal("1.0"),
    "semiannual": Decimal("0.52"),
    "quarterly": Decimal("0.265"),
    "monthly": Decimal("0.0875"),
}
MAX_ISSUE_AGE = {10: 70, 20: 60, 30: 50}
MIN_ISSUE_AGE = 18
MIN_FACE, MAX_FACE = 100_000, 2_000_000
POLICY_FEE = Decimal("60")
CENT = Decimal("0.01")


class QuoteError(ValueError):
    """The request cannot be quoted. The message is safe to show to the model."""


class RateBook:
    def __init__(self, rates_csv: Path):
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.execute(
            "CREATE TABLE rates (age_min INTEGER NOT NULL, age_max INTEGER NOT NULL, "
            "risk_class TEXT NOT NULL, term_years INTEGER NOT NULL, rate_per_1000 TEXT NOT NULL, "
            "PRIMARY KEY (age_min, risk_class, term_years))"
        )
        with open(rates_csv, newline="", encoding="utf-8") as f:
            rows = [
                (r["age_min"], r["age_max"], r["risk_class"], r["term_years"], r["rate_per_1000"])
                for r in csv.DictReader(f)
            ]
        self._db.executemany("INSERT INTO rates VALUES (?, ?, ?, ?, ?)", rows)
        self._db.commit()

    def quote(self, age: int, risk_class: str, term_years: int, face_amount: int, mode: str = "annual") -> dict:
        classes = {c.lower(): c for c in RISK_CLASSES}
        risk = classes.get(str(risk_class).strip().lower())
        if risk is None:
            raise QuoteError(f"risk_class must be one of {', '.join(RISK_CLASSES)}")
        mode = str(mode).strip().lower().replace("-", "")
        if mode not in MODAL_FACTORS:
            raise QuoteError(f"mode must be one of {', '.join(MODAL_FACTORS)}")
        try:
            age, term_years, face_amount = int(age), int(term_years), int(face_amount)
        except (TypeError, ValueError) as exc:
            raise QuoteError("age, term_years and face_amount must be whole numbers") from exc
        if term_years not in MAX_ISSUE_AGE:
            raise QuoteError("term_years must be 10, 20 or 30")
        if not MIN_FACE <= face_amount <= MAX_FACE:
            raise QuoteError(f"face_amount must be between {MIN_FACE} and {MAX_FACE}")
        if not MIN_ISSUE_AGE <= age <= MAX_ISSUE_AGE[term_years]:
            raise QuoteError(
                f"the {term_years}-year term is issued at ages {MIN_ISSUE_AGE} "
                f"through {MAX_ISSUE_AGE[term_years]}; age {age} is not eligible"
            )
        row = self._db.execute(
            "SELECT rate_per_1000 FROM rates "
            "WHERE ? BETWEEN age_min AND age_max AND risk_class = ? AND term_years = ?",
            (age, risk, term_years),
        ).fetchone()
        if row is None:
            lo, hi = self._db.execute("SELECT MIN(age_min), MAX(age_max) FROM rates").fetchone()
            raise QuoteError(f"no rate on file for age {age}; the rate table covers ages {lo} through {hi}")

        rate = Decimal(row[0])
        annual = (Decimal(face_amount) / 1000 * rate + POLICY_FEE).quantize(CENT, ROUND_HALF_UP)
        payment = (annual * MODAL_FACTORS[mode]).quantize(CENT, ROUND_HALF_UP)
        return {
            "age": age,
            "risk_class": risk,
            "term_years": term_years,
            "face_amount": face_amount,
            "mode": mode,
            "rate_per_1000": str(rate),
            "policy_fee": str(POLICY_FEE.quantize(CENT)),
            "annual_premium": str(annual),
            "payment": str(payment),
        }


QUOTE_TOOL = {
    "type": "function",
    "function": {
        "name": "quote_term_premium",
        "description": (
            "Price a Larkspur Term Life policy from the official rate table. Returns the annual "
            "premium and the payment for the chosen mode. Use it for every premium question; "
            "never estimate a premium yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "description": "Age at last birthday"},
                "risk_class": {"type": "string", "enum": list(RISK_CLASSES)},
                "term_years": {"type": "integer", "enum": [10, 20, 30]},
                "face_amount": {"type": "integer", "description": "Coverage in whole dollars"},
                "mode": {"type": "string", "enum": list(MODAL_FACTORS), "default": "annual"},
            },
            "required": ["age", "risk_class", "term_years", "face_amount"],
        },
    },
}

SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_policy_docs",
        "description": (
            "Search Larkspur policy, rider, claims, underwriting and billing documents. Returns "
            "the most relevant sections, each labelled with a section id to cite."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look for"}},
            "required": ["query"],
        },
    },
}

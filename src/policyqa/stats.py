"""Uncertainty for small evaluation sets.

With 40 questions, one question is worth 2.5 points of accuracy. These helpers
show how much of a gap between two systems could be noise.
"""

from __future__ import annotations

import math
import random
from typing import Sequence


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def bootstrap_ci(values: Sequence[float], n_resamples: int = 5000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap interval for the mean."""
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    n = len(values)
    means = [mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(n_resamples)]
    return percentile(means, 2.5), percentile(means, 97.5)


def paired_bootstrap_diff(
    a: Sequence[float], b: Sequence[float], n_resamples: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% interval. Resampling whole questions keeps the
    pairing, which matters because both systems find the same questions hard."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    diffs = [x - y for x, y in zip(a, b)]
    lo, hi = bootstrap_ci(diffs, n_resamples, seed)
    return mean(diffs), lo, hi


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Two-sided exact McNemar p-value. Only questions where exactly one of the
    two systems is right carry information about which is better."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    only_a = sum(1 for x, y in zip(a, b) if x and not y)
    only_b = sum(1 for x, y in zip(a, b) if y and not x)
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)

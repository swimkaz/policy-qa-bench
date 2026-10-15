import pytest

from policyqa.stats import bootstrap_ci, mcnemar_exact, paired_bootstrap_diff, percentile


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = [1.0] * 30 + [0.0] * 10
    lo, hi = bootstrap_ci(values)
    assert lo < 0.75 < hi and 0.55 < lo and hi < 0.92
    assert bootstrap_ci(values) == (lo, hi)
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)


def test_paired_difference():
    a = [1.0] * 8 + [0.0] * 2
    b = [1.0] * 4 + [0.0] * 6
    diff, lo, hi = paired_bootstrap_diff(a, b)
    assert diff == pytest.approx(0.4) and lo > 0


def test_mcnemar_exact():
    # 5 discordant pairs all favouring one side: 2 * (1/2)^5 = 0.0625
    assert mcnemar_exact([True] * 5, [False] * 5) == pytest.approx(0.0625)
    assert mcnemar_exact([True, False], [False, True]) == 1.0
    assert mcnemar_exact([True, True], [True, True]) == 1.0

import pytest

from policyqa.tools import QuoteError


def test_annual_premium_matches_hand_calculation(rates):
    # 35-39 Preferred 20-year rate is 1.01 per $1,000: 500 * 1.01 + 60 fee = 565.00
    q = rates.quote(35, "Preferred", 20, 500_000)
    assert (q["rate_per_1000"], q["annual_premium"], q["payment"]) == ("1.01", "565.00", "565.00")


def test_modal_payment_rounds_half_up(rates):
    # 565.00 * 0.0875 = 49.4375 -> 49.44
    assert rates.quote(35, "preferred", 20, 500_000, "monthly")["payment"] == "49.44"
    # 890.00 * 0.265 = 235.85
    assert rates.quote(28, "Preferred Plus", 30, 1_000_000, "quarterly")["payment"] == "235.85"


def test_age_band_edges(rates):
    assert rates.quote(39, "Standard", 20, 100_000)["rate_per_1000"] == "1.35"
    assert rates.quote(40, "Standard", 20, 100_000)["rate_per_1000"] == "1.95"


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(age=52, risk_class="Standard", term_years=30, face_amount=500_000), "not eligible"),
        (dict(age=62, risk_class="Standard", term_years=10, face_amount=200_000), "ages 25 through 54"),
        (dict(age=35, risk_class="Elite", term_years=20, face_amount=500_000), "risk_class"),
        (dict(age=35, risk_class="Standard", term_years=15, face_amount=500_000), "term_years"),
        (dict(age=35, risk_class="Standard", term_years=20, face_amount=50_000), "face_amount"),
        (dict(age=35, risk_class="Standard", term_years=20, face_amount=500_000, mode="weekly"), "mode"),
        (dict(age="old", risk_class="Standard", term_years=20, face_amount=500_000), "whole numbers"),
    ],
)
def test_bad_requests_raise_a_readable_error(rates, kwargs, message):
    with pytest.raises(QuoteError, match=message):
        rates.quote(**kwargs)

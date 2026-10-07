from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prodanalytics.clv import BGNBD, GammaGamma, customer_lifetime_value, rfm_summary

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


@pytest.fixture(scope="module")
def cdnow_sample_summary():
    return pd.read_csv(RAW / "cdnow_customers_summary.csv")


@pytest.fixture(scope="module")
def bgnbd(cdnow_sample_summary):
    s = cdnow_sample_summary
    return BGNBD().fit(s["frequency"], s["recency"], s["T"])


def test_bgnbd_reproduces_published_parameters(bgnbd):
    # Fader, Hardie & Lee (2005), Marketing Science: r=0.243, alpha=4.414, a=0.793, b=2.426, LL=-9582.4
    assert bgnbd.r == pytest.approx(0.243, rel=0.01)
    assert bgnbd.alpha == pytest.approx(4.414, rel=0.01)
    assert bgnbd.a == pytest.approx(0.793, rel=0.01)
    assert bgnbd.b == pytest.approx(2.426, rel=0.01)
    assert bgnbd.log_likelihood_ == pytest.approx(-9582.4, abs=0.1)


def test_bgnbd_conditional_expectation_matches_paper(bgnbd):
    # Worked example in the paper: x=2, t_x=30.43, T=38.86 -> 1.226 expected purchases in the next 39 weeks.
    assert float(bgnbd.expected_purchases(39, 2, 30.43, 38.86)) == pytest.approx(1.226, abs=0.002)


def test_prob_alive_is_one_without_repeat_purchases(bgnbd):
    assert float(bgnbd.prob_alive(0, 0, 30)) == 1.0
    assert 0 < float(bgnbd.prob_alive(5, 2, 38)) < float(bgnbd.prob_alive(5, 37, 38))


def test_gamma_gamma_reproduces_published_parameters():
    # Fader, Hardie & Lee (2005), JMR: p=6.25, q=3.74, gamma=15.44
    s = pd.read_csv(RAW / "cdnow_customers_summary_with_transactions.csv")
    gg = GammaGamma().fit(s["x"], s["zbar"])
    assert gg.p == pytest.approx(6.25, rel=0.01)
    assert gg.q == pytest.approx(3.74, rel=0.01)
    assert gg.gamma == pytest.approx(15.44, rel=0.01)


def test_rfm_summary_merges_same_day_purchases():
    tx = pd.DataFrame({
        "cid": ["a", "a", "a", "a", "b"],
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-15", "2024-01-29", "2024-01-08"]),
        "value": [10.0, 5.0, 20.0, 30.0, 7.0],
    })
    out = rfm_summary(tx, "cid", "date", "value", pd.Timestamp("2024-02-05"))
    a, b = out.loc["a"], out.loc["b"]
    assert a["frequency"] == 2
    assert a["recency"] == pytest.approx(4.0)
    assert a["T"] == pytest.approx(5.0)
    assert a["monetary"] == pytest.approx(25.0)
    assert b["frequency"] == 0 and b["monetary"] == 0


def test_rfm_summary_matches_published_cdnow_sample(cdnow_sample_summary):
    tx = pd.read_csv(RAW / "CDNOW_sample.txt", sep=r"\s+", header=None,
                     names=["master_id", "sample_id", "date", "units", "value"])
    tx["date"] = pd.to_datetime(tx["date"].astype(str), format="%Y%m%d")
    ours = rfm_summary(tx, "sample_id", "date", "value", pd.Timestamp("1997-09-30"))
    ref = cdnow_sample_summary.set_index("ID")
    joined = ours.join(ref, rsuffix="_ref")
    assert (joined["frequency"] == joined["frequency_ref"]).all()
    assert np.allclose(joined["recency"], joined["recency_ref"], atol=0.01)
    assert np.allclose(joined["T"], joined["T_ref"], atol=0.01)


def test_clv_grows_with_horizon(bgnbd):
    gg = GammaGamma(p=6.25, q=3.74, gamma=15.44)
    summary = pd.DataFrame({"frequency": [0, 3, 10], "recency": [0, 20, 37], "T": [38, 38, 38],
                            "monetary": [0, 30, 50]})
    short = customer_lifetime_value(bgnbd, gg, summary, months=6)
    long = customer_lifetime_value(bgnbd, gg, summary, months=12)
    assert (short > 0).all() and (long > short).all()
    assert long[2] > long[1] > long[0]


def test_gamma_gamma_expected_spend_shrinks_towards_population_mean():
    gg = GammaGamma(p=6.25, q=3.74, gamma=15.44)
    population_mean = 6.25 * 15.44 / 2.74            # 35.219
    assert float(gg.expected_spend(0, 0)) == pytest.approx(population_mean, rel=1e-6)
    # Hand-calculated: weight = 2.74 / (6.25*2 + 2.74) = 0.17979 -> 0.17979*35.219 + 0.82021*20 = 22.736
    assert float(gg.expected_spend(2, 20)) == pytest.approx(22.736, abs=0.01)
    # More purchases -> more weight on the customer's own average.
    assert abs(float(gg.expected_spend(50, 20)) - 20) < abs(float(gg.expected_spend(2, 20)) - 20)


# Reference values below were computed independently with the open-source `lifetimes`
# package (v0.11.3) on the same CDNOW sample, to cross-check every prediction formula.
def test_prob_alive_matches_lifetimes(bgnbd):
    ours = bgnbd.prob_alive(np.array([0, 1, 2, 5, 10]), np.array([0, 5, 30.43, 20, 37]), np.full(5, 38.86))
    expected = [1.0, 0.31491830, 0.72660914, 0.28730903, 0.90185013]
    assert ours == pytest.approx(expected, rel=1e-5)


def test_new_customer_expectation_matches_lifetimes(bgnbd):
    ours = bgnbd.expected_purchases_new_customer(np.array([10, 39, 78]))
    assert ours == pytest.approx([0.43176099, 1.19501623, 1.85797170], rel=1e-5)


def test_expected_spend_matches_lifetimes():
    s = pd.read_csv(RAW / "cdnow_customers_summary_with_transactions.csv")
    gg = GammaGamma().fit(s["x"], s["zbar"])
    ours = gg.expected_spend(np.array([1, 2, 5]), np.array([20.0, 20.0, 50.0]))
    assert ours == pytest.approx([24.62986182, 22.73176939, 48.80255453], rel=1e-5)


def test_clv_matches_lifetimes(bgnbd):
    s = pd.read_csv(RAW / "cdnow_customers_summary_with_transactions.csv")
    gg = GammaGamma().fit(s["x"], s["zbar"])
    summary = pd.DataFrame({"frequency": [1, 3, 10], "recency": [5.0, 20.0, 37.0], "T": [38.86] * 3,
                            "monetary": [20.0, 30.0, 50.0]})
    ours = customer_lifetime_value(bgnbd, gg, summary, months=12, monthly_discount=0.01)
    assert ours == pytest.approx([8.96218077, 41.02714341, 384.51877535], rel=1e-5)

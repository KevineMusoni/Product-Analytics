import numpy as np
import pytest
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep, proportions_ztest

from prodanalytics.experiment import (bayesian_ab, bootstrap_diff, bootstrap_proportion_diff, holm_adjust,
                                      minimum_detectable_effect, required_sample_size, srm_test,
                                      two_proportion_test)
from prodanalytics.peeking import simulate_peeking

# Day-7 retention counts from the Cookie Cats experiment.
X_C, N_C, X_T, N_T = 8502, 44700, 8279, 45489


def test_z_test_matches_statsmodels():
    ours = two_proportion_test(X_C, N_C, X_T, N_T)
    z, p = proportions_ztest([X_T, X_C], [N_T, N_C])
    assert ours.z == pytest.approx(z, rel=1e-9)
    assert ours.p_value == pytest.approx(p, rel=1e-9)


def test_confidence_interval_matches_statsmodels_wald():
    ours = two_proportion_test(X_C, N_C, X_T, N_T)
    lo, hi = confint_proportions_2indep(X_T, N_T, X_C, N_C, method="wald", compare="diff")
    assert ours.abs_ci == pytest.approx((lo, hi), rel=1e-9)


def test_relative_lift_interval_contains_point_estimate():
    ours = two_proportion_test(X_C, N_C, X_T, N_T)
    assert ours.rel_ci[0] < ours.rel_lift < ours.rel_ci[1]


def test_srm_matches_scipy():
    ours = srm_test(44700, 45489)
    chi2, p = stats.chisquare([44700, 45489])
    assert ours.chi2 == pytest.approx(chi2)
    assert ours.p_value == pytest.approx(p)


def test_holm_matches_statsmodels():
    pvals = [0.0016, 0.074, 0.050, 0.30]
    expected = multipletests(pvals, method="holm")[1]
    assert holm_adjust(pvals) == pytest.approx(expected)


def test_power_formulas_are_consistent():
    n = 44_700
    mde = minimum_detectable_effect(0.19, n)
    assert required_sample_size(0.19, mde) == pytest.approx(n, rel=0.03)


def test_binomial_bootstrap_agrees_with_wald_interval():
    wald = two_proportion_test(X_C, N_C, X_T, N_T).abs_ci
    _, boot, _ = bootstrap_proportion_diff(X_C, N_C, X_T, N_T, n_boot=200_000, seed=1)
    assert boot == pytest.approx(wald, abs=3e-4)


def test_bayesian_direction_agrees_with_frequentist():
    res = bayesian_ab(X_C, N_C, X_T, N_T, seed=1)
    assert res.prob_treatment_better < 0.01
    assert res.diff_credible_interval[1] < 0


def test_peeking_inflates_false_positives():
    out = simulate_peeking(n_sims=3_000, seed=7)
    assert out["fixed_horizon_false_positive_rate"] == pytest.approx(0.05, abs=0.015)
    assert out["peeking_false_positive_rate"] > 0.15


def test_relative_lift_interval_matches_statsmodels_log_ratio():
    ours = two_proportion_test(X_C, N_C, X_T, N_T)
    lo, hi = confint_proportions_2indep(X_T, N_T, X_C, N_C, method="log", compare="ratio")
    assert ours.rel_ci == pytest.approx((lo - 1, hi - 1), rel=1e-9)


def test_required_sample_size_matches_hand_calculation():
    # n = (z_0.975 + z_0.8)^2 * (p1(1-p1) + p2(1-p2)) / mde^2, worked by hand:
    # 2.801585^2 * (0.19*0.81 + 0.195*0.805) / 0.005^2 = 97,600.6 -> 97,601
    assert required_sample_size(0.19, 0.005) == 97_601
    assert required_sample_size(0.30, 0.03) == 3_760


def test_expected_loss_falls_on_the_worse_variant():
    res = bayesian_ab(X_C, N_C, X_T, N_T, seed=3)
    observed_gap = X_C / N_C - X_T / N_T            # treatment is ~0.82 pp worse
    assert res.expected_loss_treatment == pytest.approx(observed_gap, abs=2e-4)
    assert res.expected_loss_control < 1e-5


def test_bootstrap_diff_is_treatment_minus_control():
    rng = np.random.default_rng(0)
    control, treatment = rng.normal(0, 1, 2_000), rng.normal(1, 1, 2_000)
    diff, (lo, hi), _ = bootstrap_diff(control, treatment, n_boot=500, seed=0)
    assert 0.9 < diff < 1.1 and 0 < lo < diff < hi

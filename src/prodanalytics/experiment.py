"""Statistical tools for analysing product A/B experiments.

Every function here is deliberately small and explicit so the maths can be
checked line by line. Tests in ``tests/test_experiment.py`` compare the results
against reference implementations (statsmodels / scipy).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


# --------------------------------------------------------------------------- #
# Sample ratio mismatch
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SRMResult:
    n_control: int
    n_treatment: int
    expected_share_treatment: float
    observed_share_treatment: float
    chi2: float
    p_value: float

    @property
    def flagged(self) -> bool:
        """Conventional SRM alarm threshold used by large experimentation platforms."""
        return self.p_value < 0.001


def srm_test(n_control: int, n_treatment: int, expected_share_treatment: float = 0.5) -> SRMResult:
    """Chi-square goodness-of-fit test for a sample ratio mismatch.

    A significant result means the randomisation (or the logging) did not
    produce the split the experiment was designed for, so downstream effect
    estimates may be biased.
    """
    total = n_control + n_treatment
    expected = np.array([1 - expected_share_treatment, expected_share_treatment]) * total
    chi2, p = stats.chisquare([n_control, n_treatment], f_exp=expected)
    return SRMResult(n_control, n_treatment, expected_share_treatment, n_treatment / total, float(chi2), float(p))


# --------------------------------------------------------------------------- #
# Difference in conversion / retention rates
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProportionTestResult:
    rate_control: float
    rate_treatment: float
    abs_diff: float
    abs_ci: tuple[float, float]
    rel_lift: float
    rel_ci: tuple[float, float]
    z: float
    p_value: float


def two_proportion_test(x_c: int, n_c: int, x_t: int, n_t: int, alpha: float = 0.05) -> ProportionTestResult:
    """Two-sided z-test for the difference between two proportions.

    * The test statistic uses the pooled standard error (the null hypothesis
      says both groups share one rate).
    * The confidence interval for the absolute difference uses the unpooled
      standard error (no such assumption once we estimate the effect).
    * The relative lift interval is built on the log risk ratio (delta
      method), which keeps it asymmetric and correctly bounded.
    """
    p_c, p_t = x_c / n_c, x_t / n_t
    diff = p_t - p_c

    pooled = (x_c + x_t) / (n_c + n_t)
    se_pooled = np.sqrt(pooled * (1 - pooled) * (1 / n_c + 1 / n_t))
    z = diff / se_pooled
    p_value = 2 * stats.norm.sf(abs(z))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    se_unpooled = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    abs_ci = (diff - z_crit * se_unpooled, diff + z_crit * se_unpooled)

    log_rr = np.log(p_t / p_c)
    se_log_rr = np.sqrt((1 - p_t) / x_t + (1 - p_c) / x_c)
    rel_ci = (np.exp(log_rr - z_crit * se_log_rr) - 1, np.exp(log_rr + z_crit * se_log_rr) - 1)

    return ProportionTestResult(p_c, p_t, diff, abs_ci, p_t / p_c - 1, rel_ci, float(z), float(p_value))


# --------------------------------------------------------------------------- #
# Power and sample size
# --------------------------------------------------------------------------- #
def minimum_detectable_effect(p_baseline: float, n_per_group: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest absolute difference detectable with the given sample (normal approximation)."""
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    return float((z_a + z_b) * np.sqrt(2 * p_baseline * (1 - p_baseline) / n_per_group))


def required_sample_size(p_baseline: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Users needed per group to detect an absolute change of ``mde_abs``."""
    p_alt = p_baseline + mde_abs
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    var = p_baseline * (1 - p_baseline) + p_alt * (1 - p_alt)
    return int(np.ceil((z_a + z_b) ** 2 * var / mde_abs ** 2))


# --------------------------------------------------------------------------- #
# Resampling and Bayesian views of the same question
# --------------------------------------------------------------------------- #
def bootstrap_diff(control: np.ndarray, treatment: np.ndarray, stat=np.mean, n_boot: int = 10_000,
                   alpha: float = 0.05, seed: int = 0) -> tuple[float, tuple[float, float], np.ndarray]:
    """Percentile bootstrap interval for ``stat(treatment) - stat(control)``."""
    rng = np.random.default_rng(seed)
    control, treatment = np.asarray(control), np.asarray(treatment)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        c = rng.choice(control, size=control.size, replace=True)
        t = rng.choice(treatment, size=treatment.size, replace=True)
        draws[i] = stat(t) - stat(c)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return float(stat(treatment) - stat(control)), (float(lo), float(hi)), draws


def bootstrap_proportion_diff(x_c: int, n_c: int, x_t: int, n_t: int, n_boot: int = 100_000,
                              alpha: float = 0.05, seed: int = 0) -> tuple[float, tuple[float, float], np.ndarray]:
    """Bootstrap interval for a difference in proportions.

    Resampling a 0/1 column with replacement and taking the mean is the same
    as drawing a binomial count, so this is an exact and much faster
    equivalent of :func:`bootstrap_diff` for conversion-style metrics.
    """
    rng = np.random.default_rng(seed)
    draws = rng.binomial(n_t, x_t / n_t, n_boot) / n_t - rng.binomial(n_c, x_c / n_c, n_boot) / n_c
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return x_t / n_t - x_c / n_c, (float(lo), float(hi)), draws


@dataclass(frozen=True)
class BayesianResult:
    prob_treatment_better: float
    expected_loss_treatment: float
    expected_loss_control: float
    diff_credible_interval: tuple[float, float]


def bayesian_ab(x_c: int, n_c: int, x_t: int, n_t: int, prior: tuple[float, float] = (1.0, 1.0),
                draws: int = 400_000, seed: int = 0) -> BayesianResult:
    """Beta-Binomial comparison of two rates.

    Expected loss is the average amount of the metric we would give up by
    shipping a variant if it were in fact the worse one, which is often a
    more useful decision quantity than a p-value.
    """
    rng = np.random.default_rng(seed)
    a0, b0 = prior
    s_c = rng.beta(a0 + x_c, b0 + n_c - x_c, draws)
    s_t = rng.beta(a0 + x_t, b0 + n_t - x_t, draws)
    diff = s_t - s_c
    lo, hi = np.quantile(diff, [0.025, 0.975])
    return BayesianResult(
        prob_treatment_better=float((diff > 0).mean()),
        expected_loss_treatment=float(np.maximum(s_c - s_t, 0).mean()),
        expected_loss_control=float(np.maximum(s_t - s_c, 0).mean()),
        diff_credible_interval=(float(lo), float(hi)),
    )


# --------------------------------------------------------------------------- #
# Multiple comparisons
# --------------------------------------------------------------------------- #
def holm_adjust(p_values: list[float]) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * p[idx])
        adjusted[idx] = min(1.0, running_max)
    return adjusted

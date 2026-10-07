"""Simulation showing why repeatedly checking an A/B test inflates false positives.

We run many A/A tests (both groups share the same true rate, so every
"significant" result is a false positive) and compare two analysts:

* the *fixed-horizon* analyst looks once, at the planned sample size;
* the *peeking* analyst checks after every batch of users and stops as soon
  as p < alpha.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def simulate_peeking(p: float = 0.19, n_per_group: int = 20_000, looks: int = 20, n_sims: int = 2_000,
                     alpha: float = 0.05, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    checkpoints = np.linspace(n_per_group / looks, n_per_group, looks).astype(int)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    # Cumulative successes at each checkpoint for every simulated experiment.
    batch_sizes = np.diff(np.concatenate([[0], checkpoints]))
    succ_a = rng.binomial(batch_sizes, p, size=(n_sims, looks)).cumsum(axis=1)
    succ_b = rng.binomial(batch_sizes, p, size=(n_sims, looks)).cumsum(axis=1)

    n = checkpoints[None, :]
    p_a, p_b = succ_a / n, succ_b / n
    pooled = (succ_a + succ_b) / (2 * n)
    se = np.sqrt(pooled * (1 - pooled) * (2 / n))
    z = np.abs(p_b - p_a) / se

    significant = z > z_crit
    return {
        "fixed_horizon_false_positive_rate": float(significant[:, -1].mean()),
        "peeking_false_positive_rate": float(significant.any(axis=1).mean()),
        "looks": looks,
        "n_sims": n_sims,
    }

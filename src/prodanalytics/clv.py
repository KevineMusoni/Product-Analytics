"""Customer lifetime value with the BG/NBD and Gamma-Gamma models.

Implemented from the original papers rather than a library, so every step is
visible:

* Fader, Hardie & Lee (2005), "Counting your customers the easy way: an
  alternative to the Pareto/NBD model", Marketing Science 24(2).
* Fader, Hardie & Lee (2005), "RFM and CLV: using iso-value curves for
  customer base analysis", Journal of Marketing Research 42(4).

The tests check that the fitted parameters reproduce the values published in
those papers for the CDNOW sample.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.special import betaln, gammaln, hyp2f1


# --------------------------------------------------------------------------- #
# Transaction log -> recency / frequency / monetary summary
# --------------------------------------------------------------------------- #
def rfm_summary(transactions: pd.DataFrame, customer_col: str, date_col: str, value_col: str,
                observation_end: pd.Timestamp, freq_days: float = 7.0) -> pd.DataFrame:
    """Summarise a transaction log in the form the models expect.

    Purchases on the same day are merged into one purchase occasion.

    * ``frequency``  number of *repeat* purchase occasions
    * ``recency``    time from first to last purchase
    * ``T``          time from first purchase to the end of the window
    * ``monetary``   average value of the repeat purchases (0 if none)

    Times are measured in units of ``freq_days`` (weeks by default).
    """
    tx = transactions[transactions[date_col] <= observation_end]
    daily = tx.groupby([customer_col, date_col], as_index=False)[value_col].sum()
    daily = daily.sort_values([customer_col, date_col])

    first = daily.groupby(customer_col)[date_col].transform("min")
    daily["is_repeat"] = daily[date_col] > first

    g = daily.groupby(customer_col)
    out = pd.DataFrame({
        "first_purchase": g[date_col].min(),
        "last_purchase": g[date_col].max(),
        "frequency": g["is_repeat"].sum().astype(int),
    })
    repeat_value = daily[daily["is_repeat"]].groupby(customer_col)[value_col].mean()
    out["monetary"] = repeat_value.reindex(out.index).fillna(0.0)
    unit = pd.Timedelta(days=freq_days)
    out["recency"] = (out["last_purchase"] - out["first_purchase"]) / unit
    out["T"] = (observation_end - out["first_purchase"]) / unit
    return out[["frequency", "recency", "T", "monetary", "first_purchase"]]


# --------------------------------------------------------------------------- #
# BG/NBD: how many purchases will each customer make?
# --------------------------------------------------------------------------- #
@dataclass
class BGNBD:
    r: float = np.nan
    alpha: float = np.nan
    a: float = np.nan
    b: float = np.nan

    @staticmethod
    def _log_likelihood(params, x, t_x, T):
        r, alpha, a, b = params
        common = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        term1 = betaln(a, b + x) - betaln(a, b) - (r + x) * np.log(alpha + T)
        # The second term only exists for customers with at least one repeat purchase.
        x_safe = np.where(x > 0, x, 1)
        term2 = betaln(a + 1, b + x_safe - 1) - betaln(a, b) - (r + x) * np.log(alpha + t_x)
        ll = np.where(x > 0, common + np.logaddexp(term1, term2), common + term1)
        return ll

    def fit(self, frequency, recency, T) -> "BGNBD":
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))

        def nll(log_params):
            return -self._log_likelihood(np.exp(log_params), x, t_x, T).sum()

        start = np.log([1.0, max(T.mean(), 1.0), 1.0, 1.0])
        res = optimize.minimize(nll, start, method="Nelder-Mead",
                                options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-10})
        res = optimize.minimize(nll, res.x, method="L-BFGS-B")
        self.r, self.alpha, self.a, self.b = np.exp(res.x)
        self.log_likelihood_ = -res.fun
        return self

    @property
    def params(self) -> dict[str, float]:
        return {"r": self.r, "alpha": self.alpha, "a": self.a, "b": self.b}

    def prob_alive(self, frequency, recency, T):
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        x_safe = np.where(x > 0, x, 1)
        ratio = (a / (b + x_safe - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x)
        return np.where(x > 0, 1.0 / (1.0 + ratio), 1.0)

    def expected_purchases(self, t, frequency, recency, T):
        """Expected repeat purchases in the next ``t`` periods, given each customer's history."""
        x, t_x, T = (np.asarray(v, dtype=float) for v in (frequency, recency, T))
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        hyp = hyp2f1(r + x, b + x, a + b + x - 1, t / (alpha + T + t))
        numer = (a + b + x - 1) / (a - 1) * (1 - ((alpha + T) / (alpha + T + t)) ** (r + x) * hyp)
        x_safe = np.where(x > 0, x, 1)
        denom = 1 + np.where(x > 0, (a / (b + x_safe - 1)) * ((alpha + T) / (alpha + t_x)) ** (r + x), 0.0)
        return numer / denom

    def expected_purchases_new_customer(self, t):
        """Expected repeat purchases in ``(0, t]`` for a customer who has just made a first purchase."""
        r, alpha, a, b = self.r, self.alpha, self.a, self.b
        t = np.asarray(t, dtype=float)
        hyp = hyp2f1(r, b, a + b - 1, t / (alpha + t))
        return (a + b - 1) / (a - 1) * (1 - (alpha / (alpha + t)) ** r * hyp)


# --------------------------------------------------------------------------- #
# Gamma-Gamma: how much will each customer spend per purchase?
# --------------------------------------------------------------------------- #
@dataclass
class GammaGamma:
    p: float = np.nan
    q: float = np.nan
    gamma: float = np.nan

    @staticmethod
    def _log_likelihood(params, x, m):
        p, q, g = params
        return (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(g)
                + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(g + m * x))

    def fit(self, frequency, monetary) -> "GammaGamma":
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary, dtype=float)
        # The model describes positive spend among repeat customers only.
        keep = (x > 0) & (m > 0)
        x, m = x[keep], m[keep]

        def nll(log_params):
            return -self._log_likelihood(np.exp(log_params), x, m).sum()

        res = optimize.minimize(nll, np.log([1.0, 1.0, 1.0]), method="Nelder-Mead",
                                options={"maxiter": 20_000, "xatol": 1e-10, "fatol": 1e-10})
        res = optimize.minimize(nll, res.x, method="L-BFGS-B")
        self.p, self.q, self.gamma = np.exp(res.x)
        return self

    @property
    def params(self) -> dict[str, float]:
        return {"p": self.p, "q": self.q, "gamma": self.gamma}

    def expected_spend(self, frequency, monetary):
        """Expected value per transaction, shrinking each customer's average toward the population mean."""
        x = np.asarray(frequency, dtype=float)
        m = np.asarray(monetary, dtype=float)
        p, q, g = self.p, self.q, self.gamma
        population_mean = p * g / (q - 1)
        weight = (q - 1) / (p * x + q - 1)
        return weight * population_mean + (1 - weight) * m


# --------------------------------------------------------------------------- #
# Putting the two together
# --------------------------------------------------------------------------- #
def customer_lifetime_value(bgnbd: BGNBD, gg: GammaGamma, summary: pd.DataFrame, months: int = 12,
                            weeks_per_month: float = 4.345, monthly_discount: float = 0.01,
                            margin: float = 1.0) -> np.ndarray:
    """Discounted expected value of each customer's future purchases over ``months`` months.

    The expected purchase count is evaluated month by month so that later
    purchases can be discounted, then multiplied by the expected spend per
    purchase and an optional profit margin.
    """
    f, rcy, T = summary["frequency"], summary["recency"], summary["T"]
    spend = gg.expected_spend(f, summary["monetary"])
    clv = np.zeros(len(summary))
    prev = np.zeros(len(summary))
    for m in range(1, months + 1):
        cum = bgnbd.expected_purchases(m * weeks_per_month, f, rcy, T)
        clv += (cum - prev) * spend * margin / (1 + monthly_discount) ** m
        prev = cum
    return clv

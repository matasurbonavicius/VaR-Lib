"""
Christoffersen tests -- independence and conditional coverage.

Kupiec only checks the *number* of breaches. But a model can have the right
number of breaches and still be dangerous if they all cluster together (e.g. all
five breaches of the year land in one bad week). Christoffersen adds a test of
*independence*: given a breach today, is a breach tomorrow more likely?

Three statistics are produced:
  * LR_uc  -- unconditional coverage (the same idea as Kupiec).
  * LR_ind -- independence: are breaches serially independent?
  * LR_cc  -- conditional coverage = LR_uc + LR_ind, the joint test.

The independence test uses a first-order Markov chain on the breach sequence and
compares it against the null of no dependence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from varlib.backtest._chi_square import chi_square_sf


@dataclass
class ChristoffersenResult:
    """Outcome of the Christoffersen independence / conditional-coverage test."""

    n_observations: int
    n_breaches: int
    lr_unconditional: float
    lr_independence: float
    lr_conditional: float
    p_value_independence: float
    p_value_conditional: float
    reject_independence: bool
    reject_conditional: bool
    steps: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        verdict = "REJECT model" if self.reject_conditional else "model OK"
        return (
            f"ChristoffersenResult(breaches={self.n_breaches}/{self.n_observations}, "
            f"p_independence={self.p_value_independence:.4f}, "
            f"p_conditional={self.p_value_conditional:.4f}, {verdict})"
        )


def christoffersen_test(
    breaches: np.ndarray,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> ChristoffersenResult:
    """
    Run the Christoffersen independence and conditional-coverage tests.

    Parameters
    ----------
    breaches
        1-D array of 0/1 breach flags, in time order.
    confidence
        The VaR confidence level, e.g. 0.99.
    significance
        Significance level for the reject/accept decisions.
    """
    steps: dict[str, Any] = {}

    breaches = np.asarray(breaches, dtype=int)
    n = int(breaches.size)
    if n < 2:
        raise ValueError("Need at least two observations for a Markov test.")
    steps["n_observations"] = n
    steps["n_breaches"] = int(np.sum(breaches))

    # Step 1: count the four transition types between consecutive days.
    # n_ij = number of times state i was followed by state j (0 = no breach).
    prev = breaches[:-1]
    curr = breaches[1:]
    n00 = int(np.sum((prev == 0) & (curr == 0)))
    n01 = int(np.sum((prev == 0) & (curr == 1)))
    n10 = int(np.sum((prev == 1) & (curr == 0)))
    n11 = int(np.sum((prev == 1) & (curr == 1)))
    steps["transitions"] = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}

    # Step 2: transition probabilities under the Markov (dependent) model.
    # pi_01 = P(breach today | no breach yesterday); pi_11 = P(breach | breach).
    pi_01 = _safe_ratio(n01, n00 + n01)
    pi_11 = _safe_ratio(n11, n10 + n11)
    # Pooled breach rate under the independence (no-dependence) model.
    pi = _safe_ratio(n01 + n11, n00 + n01 + n10 + n11)
    steps["pi_01"] = pi_01
    steps["pi_11"] = pi_11
    steps["pi_pooled"] = pi

    # Step 3: LR_ind compares the dependent model against the independent one.
    log_lik_dependent = (
        _xlogp(n01, pi_01) + _xlogp(n00, 1 - pi_01)
        + _xlogp(n11, pi_11) + _xlogp(n10, 1 - pi_11)
    )
    log_lik_independent = (
        _xlogp(n01 + n11, pi) + _xlogp(n00 + n10, 1 - pi)
    )
    lr_ind = -2.0 * (log_lik_independent - log_lik_dependent)
    lr_ind = max(lr_ind, 0.0)
    steps["lr_independence"] = lr_ind

    # Step 4: LR_uc is the unconditional-coverage statistic (Kupiec form).
    expected_rate = 1.0 - confidence
    x = n01 + n11        # total breaches among the transition pairs
    n_pairs = n - 1
    observed_rate = _safe_ratio(x, n_pairs)
    log_lik_null = _xlogp(x, expected_rate) + _xlogp(n_pairs - x, 1 - expected_rate)
    log_lik_obs = _xlogp(x, observed_rate) + _xlogp(n_pairs - x, 1 - observed_rate)
    lr_uc = -2.0 * (log_lik_null - log_lik_obs)
    lr_uc = max(lr_uc, 0.0)
    steps["lr_unconditional"] = lr_uc

    # Step 5: LR_cc, the conditional-coverage test, is the sum of the two.
    # It is chi-square with 2 degrees of freedom (1 for coverage, 1 for independence).
    lr_cc = lr_uc + lr_ind
    steps["lr_conditional"] = lr_cc

    # Step 6: p-values and reject decisions.
    p_ind = chi_square_sf(lr_ind, df=1)
    p_cc = chi_square_sf(lr_cc, df=2)
    steps["p_value_independence"] = p_ind
    steps["p_value_conditional"] = p_cc

    return ChristoffersenResult(
        n_observations=n,
        n_breaches=int(np.sum(breaches)),
        lr_unconditional=lr_uc,
        lr_independence=lr_ind,
        lr_conditional=lr_cc,
        p_value_independence=p_ind,
        p_value_conditional=p_cc,
        reject_independence=bool(p_ind < significance),
        reject_conditional=bool(p_cc < significance),
        steps=steps,
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator/denominator, or 0.0 if the denominator is zero."""
    return numerator / denominator if denominator > 0 else 0.0


def _xlogp(count: float, p: float) -> float:
    """
    count * ln(p), with the convention 0 * ln(0) = 0.

    Used to build binomial log-likelihoods. If count is positive but p is zero
    the term is genuinely -inf, but in that case count must also be zero by
    construction, so the convention is safe here.
    """
    if count <= 0:
        return 0.0
    if p <= 0:
        return 0.0
    return count * math.log(p)

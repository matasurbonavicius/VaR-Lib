"""
Kupiec Proportion-of-Failures (POF) test -- unconditional coverage.

Did the VaR get breached about as often as it should have? 
At 99% confidence we expect a breach on about 1% of days. 
If we see far more or far fewer, the model is mis-calibrated.

The result of the test is to accept if

PValuePOF<F(TestLevel)

and reject otherwise, where F is the cumulative distribution 
of a chi-square variable with 1 degree of freedom.

Source see: https://www.mathworks.com/help/risk/varbacktest.pof.html
Based on: Kupiec, P. "Techniques for Verifying the Accuracy of Risk Management Models." Journal of Derivatives. Vol. 3, 1995, pp. 73 – 84.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import binom, chi2


@dataclass
class KupiecResult:
    """Outcome of the Kupiec POF test."""

    n_observations: int
    n_breaches: int
    expected_rate: float
    observed_rate: float
    statistic: float
    p_value: float
    reject_at_5pct: bool
    steps: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        verdict = "REJECT model" if self.reject_at_5pct else "model OK"
        return (
            f"KupiecResult(breaches={self.n_breaches}/{self.n_observations}, "
            f"observed={self.observed_rate:.4f}, expected={self.expected_rate:.4f}, "
            f"p={self.p_value:.4f}, {verdict})"
        )


def kupiec_pof_test(
    breaches: np.ndarray,
    confidence: float = 0.99,
    significance: float = 0.05,
) -> KupiecResult:
    """
    Run the Kupiec POF test on a sequence of breach flags.

    Parameters
    ----------
    breaches
        1-D array of 0/1 flags. 1 = the realised loss exceeded the VaR forecast.
    confidence
        The VaR confidence level the breaches were generated at, e.g. 0.99.
    significance
        Test significance level for the reject/accept decision, e.g. 0.05.
    """
    steps: dict[str, Any] = {}

    breaches = np.asarray(breaches, dtype=float)
    n = int(breaches.size)
    if n == 0:
        raise ValueError("Need at least one observation.")

    # Step 1: count the breaches and the observed breach rate.
    x = int(np.sum(breaches))            # number of breaches
    expected_rate = 1.0 - confidence     # p: the rate the model implies
    observed_rate = x / n                # p_hat: the rate we actually saw
    steps["n_observations"] = n
    steps["n_breaches"] = x
    steps["expected_rate"] = expected_rate
    steps["observed_rate"] = observed_rate

    # Binomial log-likelihood of seeing x breaches in n days if the true breach
    # rate were p: log P(X=x) for X ~ Binomial(n, p). We score it at two rates
    # and compare. scipy.stats.binom.logpmf gives this directly.
    #
    # Step 2: log-likelihood under the null (true rate = expected_rate).
    log_lik_null = float(binom.logpmf(x, n, expected_rate))
    steps["log_lik_null"] = log_lik_null

    # Step 3: log-likelihood under the alternative (true rate = observed_rate).
    log_lik_alt = float(binom.logpmf(x, n, observed_rate))
    steps["log_lik_alt"] = log_lik_alt

    # Step 4: the likelihood-ratio statistic, chi-square with 1 dof.
    statistic = -2.0 * (log_lik_null - log_lik_alt)
    statistic = max(statistic, 0.0)  # guard tiny negative from rounding
    steps["statistic"] = statistic

    # Step 5: the p-value is the chi-square (df=1) upper-tail probability.
    p_value = float(chi2.sf(statistic, df=1))
    steps["p_value"] = p_value

    # Step 6: reject the model if the p-value is below the significance level.
    reject = bool(p_value < significance)
    steps["reject_at_5pct"] = reject

    return KupiecResult(
        n_observations=n,
        n_breaches=x,
        expected_rate=expected_rate,
        observed_rate=observed_rate,
        statistic=statistic,
        p_value=p_value,
        reject_at_5pct=reject,
        steps=steps,
    )

"""
Dynamic Quantile test (Engle & Manganelli) -- the modern conditional-coverage test.

Kupiec checks only the *number* of breaches. The Dynamic Quantile (DQ) test adds
the independence side: it asks whether the breaches can be *predicted* from
anything known the day before -- their own recent history, or the VaR's own level
-- so a model with the right average breach count but clustered or VaR-correlated
breaches still fails.

The idea. Define the centred breach (the "hit"):

    Hit_t = I(loss_t > VaR_t) - (1 - confidence)

Under a correct model the breach probability is exactly ``1 - confidence`` every
day regardless of history, so ``Hit_t`` has mean zero and is unpredictable. We
test that by regressing ``Hit_t`` on a constant, several of its own lags, and the
contemporaneous VaR forecast, then asking whether *all* those coefficients are
jointly zero. If any of them matters, the breaches are forecastable -- the model
is mis-specified, even if it has the right average breach count.

The statistic (Engle-Manganelli 2004) is

    DQ = beta_hat' (X'X) beta_hat / ( q * (1 - q) ),     q = 1 - confidence,

which is chi-square distributed with ``k`` degrees of freedom, ``k`` = number of
regressors (constant + lags + VaR). It catches both serial clustering and
breaches that track the VaR *level* -- a single regression-based test of
independence and correct coverage together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from varlib.backtest._chi_square import chi_square_sf


@dataclass
class DynamicQuantileResult:
    """Outcome of the Engle-Manganelli Dynamic Quantile test."""

    n_observations: int
    n_breaches: int
    n_lags: int
    n_regressors: int
    statistic: float
    p_value: float
    reject: bool
    steps: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        verdict = "REJECT model" if self.reject else "model OK"
        return (
            f"DynamicQuantileResult(breaches={self.n_breaches}/{self.n_observations}, "
            f"DQ={self.statistic:.3f}, p={self.p_value:.4f}, {verdict})"
        )


def dynamic_quantile_test(
    breaches: np.ndarray,
    confidence: float = 0.99,
    var_forecasts: Optional[np.ndarray] = None,
    n_lags: int = 4,
    significance: float = 0.05,
) -> DynamicQuantileResult:
    """
    Run the Engle-Manganelli Dynamic Quantile test on a breach sequence.

    Parameters
    ----------
    breaches
        1-D array of 0/1 breach flags, in time order.
    confidence
        The VaR confidence level the breaches were generated at, e.g. 0.99.
    var_forecasts
        Optional aligned VaR forecast series. When given, the contemporaneous VaR
        is included as a regressor (the full DQ test); breaches that track the VaR
        level are then detectable. When omitted, the test uses only the constant
        and the lagged hits.
    n_lags
        Number of lagged hits to include as regressors (Engle-Manganelli use 4).
    significance
        Significance level for the reject/accept decision.
    """
    steps: dict[str, Any] = {}

    breaches = np.asarray(breaches, dtype=float)
    n = int(breaches.size)
    if n_lags < 1:
        raise ValueError("n_lags must be >= 1.")
    if n <= n_lags + 2:
        raise ValueError(
            f"Need more than n_lags+2={n_lags + 2} observations, got {n}."
        )
    steps["n_observations"] = n
    steps["n_breaches"] = int(np.sum(breaches))
    steps["n_lags"] = n_lags

    q = 1.0 - confidence
    steps["expected_rate"] = q

    # Step 1: the centred hit series. Under a correct model E[Hit_t] = 0.
    hits = breaches - q
    steps["hits_mean"] = float(np.mean(hits))

    # Step 2: build the regression. Row t (for t = n_lags .. n-1) predicts Hit_t
    # from a constant, the previous `n_lags` hits, and -- if supplied -- the
    # contemporaneous VaR forecast (information available when the forecast was
    # made). We drop the first `n_lags` rows that have no full lag history.
    rows = []
    targets = []
    use_var = var_forecasts is not None
    if use_var:
        var_forecasts = np.asarray(var_forecasts, dtype=float)
        if var_forecasts.shape != breaches.shape:
            raise ValueError("var_forecasts and breaches must be the same length.")
    for t in range(n_lags, n):
        row = [1.0]                                  # constant
        row.extend(hits[t - n_lags:t][::-1])         # Hit_{t-1} .. Hit_{t-lags}
        if use_var:
            row.append(float(var_forecasts[t]))      # contemporaneous VaR
        rows.append(row)
        targets.append(hits[t])
    X = np.asarray(rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    k = X.shape[1]
    steps["n_regressors"] = k
    steps["includes_var_level"] = use_var

    # Step 3: ordinary-least-squares coefficients, beta = (X'X)^-1 X'y. We solve
    # the normal equations directly; a tiny ridge guards the rare singular case
    # (e.g. no breaches at all, so every hit equals -q and the lags are collinear).
    xtx = X.T @ X
    xty = X.T @ y
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.solve(xtx + 1e-10 * np.eye(k), xty)
    steps["coefficients"] = beta

    # Step 4: the DQ statistic. Under the null all coefficients are zero, and
    #   DQ = beta' (X'X) beta / (q (1 - q))  ~  chi-square(k).
    statistic = float(beta @ xtx @ beta / (q * (1.0 - q)))
    statistic = max(statistic, 0.0)
    steps["statistic"] = statistic
    # Keep X'X so a chart can split the statistic into per-regressor
    # contributions (see varlib.plotting.dq_chart) without re-fitting.
    steps["xtx"] = xtx

    # Step 5: the p-value (chi-square upper tail with k dof) and the decision.
    p_value = chi_square_sf(statistic, df=k)
    steps["p_value"] = p_value
    reject = bool(p_value < significance)
    steps["reject"] = reject

    return DynamicQuantileResult(
        n_observations=n,
        n_breaches=int(np.sum(breaches)),
        n_lags=n_lags,
        n_regressors=k,
        statistic=statistic,
        p_value=p_value,
        reject=reject,
        steps=steps,
    )

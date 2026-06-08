"""
Parametric Ornstein-Uhlenbeck VaR -- VaR for a mean-reverting series.

Brownian motion assumes returns wander with no memory. Many financial series
(spreads, rates, volatility, pairs) instead pull back towards a long-run level:
a big move tends to be partly reversed. The Ornstein-Uhlenbeck (OU) process
captures this.

The OU process in discrete time is exactly an AR(1) regression:

    x_t = a + b * x_{t-1} + e_t

From the fitted (a, b) and the residual volatility we recover the OU parameters
(mean-reversion speed, long-run mean, diffusion volatility) and then read the
one-period VaR off the *conditional* normal distribution of the next step.

Because the next step is conditional on where we are now, the OU VaR depends on
the most recent observation, not just on the unconditional volatility.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from varlib.base import VarModel
from varlib.models.parametric_brownian import normal_pdf, normal_quantile


class OuParameters(NamedTuple):
    """Estimated OU parameters."""

    kappa: float       # speed of mean reversion (per period)
    theta: float       # long-run mean level
    sigma: float       # diffusion volatility (per sqrt period)
    b: float           # the AR(1) slope, kept for transparency
    last_value: float  # the most recent observation we condition on


class ParametricOuVar(VarModel):
    """Mean-reverting (Ornstein-Uhlenbeck) parametric VaR."""

    method_name = "parametric_ou"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        params: OuParameters | None = None,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        self.params_override = params

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        return parametric_ou_var_es(
            returns,
            self.confidence,
            self.params_override,
            steps,
            horizon=self.horizon,
        )


def parametric_ou_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    params: OuParameters | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Compute OU VaR and ES at `horizon`, as positive loss fractions.

    The VaR is for the loss over the holding period, so we model the
    *cumulative* return over the next h steps,

        S_h = x_{t+1} + x_{t+2} + ... + x_{t+h},

    where each step follows the fitted AR(1)  x_k = a + b*x_{k-1} + e,  e ~ N(0, sigma^2).
    This is the quantity the backtest actually realises (the summed h-day return),
    so the forecast and the outcome are on the same footing. Modelling only the
    single value x_{t+h} -- the marginal h steps ahead -- would describe one day's
    return, not the h-day holding-period loss, and would badly under-state risk.

    S_h is Normal (a sum of Normals) with closed-form mean and variance:

        E[S_h]   = sum_{k=1..h} ( theta + b**k * (x0 - theta) )
        Var[S_h] = sigma**2 * sum_{j=1..h} ( (1 - b**(h-j+1)) / (1 - b) )**2

    The variance expression is the sum of squared shock-loadings: shock e_j feeds
    into every later step, contributing the geometric series (1 - b^{h-j+1})/(1-b).
    The sign of the autocorrelation b then decides how multi-day risk accumulates:
    a persistent series (b > 0) compounds its shocks, so Var[S_h] grows *faster*
    than the h * sigma^2 an i.i.d. series gives; an oscillating one (b < 0) sees
    shocks partly cancel, so it grows *slower*. Either way the sqrt-of-time rule
    (which assumes b == 0) gets the multi-day scaling wrong. For ``horizon == 1``
    this reduces to mean ``theta + b*(x0-theta)`` and variance ``sigma**2`` -- the
    loss on the next single return.

    Parameters
    ----------
    params
        Optional pre-calibrated OU parameters. If omitted they are estimated
        from `returns` via the AR(1) regression in `estimate_ou_parameters`.
    """
    if steps is None:
        steps = {}

    returns = np.asarray(returns, dtype=float)
    horizon = int(horizon)

    # Step 1: get the OU parameters, either by estimation or from the override.
    if params is None:
        params = estimate_ou_parameters(returns, steps)
        steps["params_source"] = "estimated"
    else:
        steps["params_source"] = "override"
    steps["kappa"] = params.kappa
    steps["theta"] = params.theta
    steps["sigma_ou"] = params.sigma
    if horizon > 1:
        steps["horizon"] = horizon

    b, theta, sigma, x0 = (
        params.b, params.theta, params.sigma, params.last_value
    )
    k = np.arange(1, horizon + 1)

    # Step 2: the expected cumulative return E[S_h]. Each step k reverts toward
    # theta by b**k from the current value x0; we sum those h conditional means.
    step_means = theta + b ** k * (x0 - theta)
    expected_sum = float(np.sum(step_means))
    steps["expected_sum"] = expected_sum

    # Step 3: the variance of the cumulative sum. Shock e_j (j = 1..h) loads onto
    # steps j..h with weights b**0, b**1, ..., b**(h-j), summing to the geometric
    # series (1 - b**(h-j+1))/(1-b). The total variance is sigma^2 times the sum
    # of the squared loadings. (When b -> 1 this tends to sigma^2 * sum(k^2)-style
    # growth; for |b| < 1 reversion damps it below h*sigma^2.)
    if abs(1.0 - b) < 1e-12:
        loadings = np.full(horizon, float(horizon))      # b -> 1 limit
    else:
        j = np.arange(1, horizon + 1)
        loadings = (1.0 - b ** (horizon - j + 1)) / (1.0 - b)
    variance_sum = float(sigma * sigma * np.sum(loadings * loadings))
    sigma_sum = float(np.sqrt(variance_sum))
    steps["sigma_sum"] = sigma_sum

    # Step 4: left-tail standard-normal quantile at (1 - confidence).
    tail_probability = 1.0 - confidence
    z = normal_quantile(tail_probability)
    steps["z_score"] = z

    # Step 5: the cumulative return at the tail; VaR is the loss (its negative).
    return_at_quantile = expected_sum + z * sigma_sum
    steps["return_at_quantile"] = float(return_at_quantile)
    var = float(-return_at_quantile)
    steps["var"] = var

    # Step 6: ES is the Gaussian tail-average loss of the cumulative return.
    # Mean cumulative return in the tail is E[S_h] - sigma_sum * pdf(z) / tail,
    # so the average loss (its negative) is below.
    pdf_z = normal_pdf(z)
    steps["pdf_z"] = pdf_z
    es = float(-expected_sum + sigma_sum * pdf_z / tail_probability)
    steps["es"] = es

    return var, es


def parametric_ou_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    params: OuParameters | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> float:
    """Convenience wrapper returning only the OU VaR."""
    var, _ = parametric_ou_var_es(returns, confidence, params, steps, horizon)
    return var


def estimate_ou_parameters(
    series: np.ndarray,
    steps: dict[str, Any] | None = None,
) -> OuParameters:
    """
    Estimate OU parameters by fitting the AR(1) regression x_t = a + b x_{t-1}.

    Returns
    -------
    OuParameters
        kappa, theta, sigma, plus the raw slope b and the last value.
    """
    if steps is None:
        steps = {}

    series = np.asarray(series, dtype=float)
    if series.size < 3:
        raise ValueError("Need at least 3 observations to fit an AR(1) model.")

    # Step 1: line up each value against the previous value.
    x_prev = series[:-1]
    x_curr = series[1:]
    steps["x_prev"] = x_prev
    steps["x_curr"] = x_curr

    # Step 2: ordinary least squares for slope b and intercept a.
    # b = cov(prev, curr) / var(prev); a = mean(curr) - b * mean(prev).
    mean_prev = float(np.mean(x_prev))
    mean_curr = float(np.mean(x_curr))
    cov = float(np.mean((x_prev - mean_prev) * (x_curr - mean_curr)))
    var_prev = float(np.var(x_prev))
    b = cov / var_prev
    a = mean_curr - b * mean_prev
    steps["ar1_slope_b"] = b
    steps["ar1_intercept_a"] = a

    # Step 3: residuals of the regression, and their standard deviation.
    fitted = a + b * x_prev
    residuals = x_curr - fitted
    residual_sigma = float(np.std(residuals, ddof=2))  # 2 params estimated
    steps["residual_sigma"] = residual_sigma

    # Step 4: map the AR(1) coefficients to OU parameters.
    # b = exp(-kappa)  =>  kappa = -ln(b)   (requires 0 < b < 1)
    # theta = a / (1 - b)
    # sigma = residual_sigma  (already the one-step diffusion scale)
    if not 0.0 < b < 1.0:
        # No mean reversion detected; fall back to a non-reverting fit so the
        # model still returns a sensible (unconditional) number.
        kappa = 0.0
        theta = mean_curr
    else:
        kappa = float(-np.log(b))
        theta = float(a / (1.0 - b))
    steps["kappa"] = kappa
    steps["theta"] = theta

    return OuParameters(
        kappa=kappa,
        theta=theta,
        sigma=residual_sigma,
        b=b,
        last_value=float(series[-1]),
    )

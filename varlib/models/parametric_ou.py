"""
Parametric Ornstein-Uhlenbeck VaR -- VaR for a mean-reverting series.

Brownian motion assumes returns wander with no memory. Many financial series
(spreads, rates, volatility, pairs) instead pull back towards a long-run level:
a big move tends to be partly reversed. The Ornstein-Uhlenbeck (OU) process
captures this.

The OU process in discrete time is exactly an AR(1) regression:

    x_t = a + b * x_{t-1} + e_t

From the fitted (a, b) and the residual volatility we recover the OU parameters
(mean-reversion speed, long-run mean, diffusion volatility) and then read the VaR
off a *simulation*: we iterate the fitted AR(1) recursion forward `horizon` steps
many times, sum each path into one h-day return, and take the VaR and ES of the
simulated sample with the Historical definition -- the same estimate-by-Monte-
Carlo pattern as the jump-diffusion model.

Because the path starts at the most recent observation, the OU VaR depends on
where we are now, not just on the unconditional volatility. And because the
simulation is the AR(1) process itself, the multi-day risk scales by the
process's own autocorrelation (persistence amplifies it, oscillation damps it),
not by the sqrt-of-time rule.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from varlib.base import VarModel
from varlib.models.historical import historical_var, historical_es


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
        n_simulations: int = 50_000,
        seed: int = 0,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        self.params_override = params
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        return parametric_ou_var_es(
            returns,
            self.confidence,
            self.params_override,
            self.n_simulations,
            self.seed,
            steps,
            horizon=self.horizon,
        )


def parametric_ou_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    params: OuParameters | None = None,
    n_simulations: int = 50_000,
    seed: int = 0,
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
    so the forecast and the outcome are on the same footing.

    We get the distribution of S_h by *simulation*: starting every path at the
    last observed value x0, we iterate the AR(1) recursion forward h steps for
    `n_simulations` paths and sum each path. The VaR and ES are then read off the
    simulated h-day sample with the Historical definition. Because the simulated
    paths follow the AR(1) process itself, the multi-day risk scales by the
    process's own autocorrelation: a persistent series (b > 0) compounds its
    shocks and is riskier than i.i.d.; an oscillating one (b < 0) partly cancels
    and is less risky -- neither is the sqrt-of-time rule (which assumes b == 0).
    For ``horizon == 1`` each path is a single step, the loss on the next return.

    Parameters
    ----------
    params
        Optional pre-calibrated OU parameters. If omitted they are estimated
        from `returns` via the AR(1) regression in `estimate_ou_parameters`.
    n_simulations, seed
        Number of Monte Carlo paths and the RNG seed (for reproducibility).
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
    # The AR(1) intercept implied by the fitted (b, theta): theta = a / (1 - b).
    a = theta * (1.0 - b)

    # Step 2: simulate the cumulative h-day return S_h. Start every path at x0 and
    # iterate the AR(1) recursion x_k = a + b*x_{k-1} + e forward h steps, summing
    # each path into one h-day return. This is just the process's own definition.
    rng = np.random.default_rng(seed)
    steps["seed"] = int(seed)
    steps["n_simulations"] = int(n_simulations)
    x = np.full(n_simulations, x0, dtype=float)
    cumulative = np.zeros(n_simulations, dtype=float)
    for _ in range(horizon):
        x = a + b * x + rng.normal(0.0, sigma, size=n_simulations)
        cumulative += x
    steps["simulated_returns"] = cumulative

    # Step 3: VaR and ES off the simulated sample, via the Historical definition
    # (VaR = loss quantile, ES = average loss beyond it) -- same as every model.
    var = historical_var(cumulative, confidence)
    steps["var"] = var
    es = historical_es(cumulative, confidence, var)
    steps["es"] = es

    return var, es


def parametric_ou_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    params: OuParameters | None = None,
    n_simulations: int = 50_000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> float:
    """Convenience wrapper returning only the OU VaR."""
    var, _ = parametric_ou_var_es(
        returns, confidence, params, n_simulations, seed, steps, horizon
    )
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

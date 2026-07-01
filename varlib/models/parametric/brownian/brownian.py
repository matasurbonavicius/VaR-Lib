"""
Parametric Brownian-motion VaR

We assume returns are driven by a Brownian motion, i.e. they are normally
distributed with a constant mean (drift) and standard deviation (volatility).

We turn that assumption into a VaR by *simulation*: draw many returns from the
fitted Normal and read the VaR and ES straight off the simulated sample with the
Historical definition. This is the same "estimate by Monte Carlo" pattern as the
jump-diffusion model, so every model in the library agrees on what VaR/ES mean.
(For this Gaussian case there is also an exact closed form; simulation matches it
up to a small, seed-controlled sampling error.)

The two helpers ``normal_pdf`` / ``normal_quantile`` live here because the EWMA
model -- which stays a faithful closed-form transcription of RiskMetrics -- still
imports them.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from varlib.base import VarModel
from varlib.models.non_parametric.historical.historical import historical_var, historical_es


class ParametricBrownianVar(VarModel):
    """Gaussian (variance-covariance) VaR."""

    method_name = "parametric_brownian"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        mu: float | None = None,
        sigma: float | None = None,
        n_simulations: int = 50_000,
        seed: int = 0,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        self.mu_override = mu
        self.sigma_override = sigma
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        return parametric_brownian_var_es(
            returns,
            self.confidence,
            self.mu_override,
            self.sigma_override,
            self.n_simulations,
            self.seed,
            steps,
            horizon=self.horizon,
        )


def parametric_brownian_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    mu: float | None = None,
    sigma: float | None = None,
    n_simulations: int = 50_000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Compute Gaussian VaR and ES at `horizon`, as positive loss fractions.

    Under the Brownian assumption returns are i.i.d. Normal(mu, sigma^2), so the
    h-day return is Normal with mean ``h*mu`` and standard deviation
    ``sqrt(h)*sigma`` (independent increments add their means and variances). We
    simulate `n_simulations` draws from that h-day Normal and take the VaR and ES
    of the simulated sample using the Historical definition -- the same pattern as
    the jump-diffusion model. For ``horizon == 1`` this is the ordinary one-day
    Gaussian VaR. A fixed `seed` makes the result reproducible.

    Parameters
    ----------
    mu, sigma
        Optional overrides for the *one-day* drift and volatility. If omitted
        they are estimated from `returns` (sample mean and sample std).
    n_simulations, seed
        Number of Monte Carlo draws and the RNG seed (for reproducibility).
    """
    if steps is None:
        steps = {}

    returns = np.asarray(returns, dtype=float)
    horizon = int(horizon)

    # Step 1: estimate (or accept) the one-day drift mu.
    if mu is None:
        mu = float(np.mean(returns))
        steps["mu_source"] = "estimated"
    else:
        mu = float(mu)
        steps["mu_source"] = "override"
    steps["mu"] = mu

    # Step 2: estimate (or accept) the one-day volatility sigma.
    if sigma is None:
        sigma = float(np.std(returns, ddof=1))
        steps["sigma_source"] = "estimated"
    else:
        sigma = float(sigma)
        steps["sigma_source"] = "override"
    steps["sigma"] = sigma

    # Step 3: project to the h-day distribution. Independent Normal increments
    # add their means and variances, so the h-day mean is h*mu and the h-day
    # standard deviation is sqrt(h)*sigma.
    mu_h = horizon * mu
    sigma_h = float(np.sqrt(horizon)) * sigma
    if horizon > 1:
        steps["horizon"] = horizon
        steps["mu_horizon"] = float(mu_h)
        steps["sigma_horizon"] = float(sigma_h)

    # Step 4: simulate h-day returns from that Normal, then read VaR and ES off
    # the simulated sample with the Historical definition (VaR = loss quantile,
    # ES = average loss beyond it). Same estimate-by-simulation pattern as every
    # other model, so they all agree on what VaR/ES mean.
    rng = np.random.default_rng(seed)
    steps["seed"] = int(seed)
    steps["n_simulations"] = int(n_simulations)
    simulated_returns = rng.normal(mu_h, sigma_h, size=n_simulations)
    steps["simulated_returns"] = simulated_returns

    var = historical_var(simulated_returns, confidence)
    steps["var"] = var
    es = historical_es(simulated_returns, confidence, var)
    steps["es"] = es

    return var, es


def parametric_brownian_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    mu: float | None = None,
    sigma: float | None = None,
    n_simulations: int = 50_000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> float:
    """Convenience wrapper returning only the Gaussian VaR."""
    var, _ = parametric_brownian_var_es(
        returns, confidence, mu, sigma, n_simulations, seed, steps, horizon
    )
    return var


def normal_pdf(x: float) -> float:
    """Standard-normal probability density at x."""
    return float(norm.pdf(x))


def normal_quantile(p: float) -> float:
    """Inverse of the standard-normal CDF (the quantile / probit function)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    return float(norm.ppf(p))

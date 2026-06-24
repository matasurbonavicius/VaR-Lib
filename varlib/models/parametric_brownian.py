"""
Parametric Brownian-motion VaR -- the classic variance-covariance method.

We assume returns are driven by a Brownian motion, i.e. they are normally
distributed with a constant mean (drift) and standard deviation (volatility).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel


class ParametricBrownianVar(VarModel):
    """Gaussian (variance-covariance) VaR."""

    method_name = "parametric_brownian"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        mu: float | None = None,
        sigma: float | None = None,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        self.mu_override = mu
        self.sigma_override = sigma

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        return parametric_brownian_var_es(
            returns,
            self.confidence,
            self.mu_override,
            self.sigma_override,
            steps,
            horizon=self.horizon,
        )


def parametric_brownian_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    mu: float | None = None,
    sigma: float | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Compute Gaussian VaR and ES at `horizon`, as positive loss fractions.

    Under the Brownian assumption returns are i.i.d. Normal(mu, sigma^2), so the
    h-day return is Normal with mean ``h*mu`` and standard deviation
    ``sqrt(h)*sigma`` (independent increments add their means and variances).
    The closed forms are then evaluated on that h-day distribution:
        VaR = -(mu_h + z * sigma_h)
        ES  = -mu_h + sigma_h * pdf(z) / (1 - confidence)
    where ``mu_h = h*mu``, ``sigma_h = sqrt(h)*sigma`` and z is the standard-
    normal quantile at (1 - confidence). For ``horizon == 1`` this is the
    ordinary one-day Gaussian VaR.

    Parameters
    ----------
    mu, sigma
        Optional overrides for the *one-day* drift and volatility. If omitted
        they are estimated from `returns` (sample mean and sample std).
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
    sigma_h = np.sqrt(horizon) * sigma
    if horizon > 1:
        steps["horizon"] = horizon
        steps["mu_horizon"] = float(mu_h)
        steps["sigma_horizon"] = float(sigma_h)

    # Step 4: find the left-tail quantile of the standard normal.
    # For 99% confidence we want the 1% point, which is negative.
    tail_probability = 1.0 - confidence
    steps["tail_probability"] = tail_probability
    z = normal_quantile(tail_probability)
    steps["z_score"] = z

    # Step 5: the return at that quantile is mu_h + z * sigma_h (z is negative).
    return_at_quantile = mu_h + z * sigma_h
    steps["return_at_quantile"] = float(return_at_quantile)

    # Step 6: VaR is the loss, i.e. the negative of that return.
    var = float(-return_at_quantile)
    steps["var"] = var

    # Step 7: ES is the average loss in the tail beyond the VaR. For a normal,
    # the average return in the left tail is mu_h - sigma_h * pdf(z) / tail_prob,
    # so the average loss (its negative) is -mu_h + sigma_h * pdf(z) / tail_prob.
    pdf_z = normal_pdf(z)
    steps["pdf_z"] = pdf_z
    es = float(-mu_h + sigma_h * pdf_z / tail_probability)
    steps["es"] = es

    return var, es


def parametric_brownian_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    mu: float | None = None,
    sigma: float | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> float:
    """Convenience wrapper returning only the Gaussian VaR."""
    var, _ = parametric_brownian_var_es(returns, confidence, mu, sigma, steps, horizon)
    return var


def normal_pdf(x: float) -> float:
    """Standard-normal probability density at x."""
    return float(np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi))


def normal_quantile(p: float) -> float:
    """
    Inverse of the standard-normal CDF (the quantile / probit function).

    Implemented with Peter Acklam's rational approximation, which is accurate to
    roughly 1e-9 across (0, 1). This keeps the library free of scipy.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients for the rational approximation.
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00]

    # The approximation uses three regions: two tails and a central band.
    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = np.sqrt(-2.0 * np.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = np.sqrt(-2.0 * np.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)

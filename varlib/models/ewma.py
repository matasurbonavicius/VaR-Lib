"""
EWMA / RiskMetrics VaR -- volatility that remembers recent days more.

The single most-used parametric VaR in practice. The Gaussian model estimates
one volatility from the whole window, so a calm day a year ago counts exactly as
much as yesterday's crash. Markets do not behave that way: volatility *clusters*
-- turbulent days follow turbulent days. EWMA (exponentially weighted moving
average, J.P. Morgan's RiskMetrics) fixes this by weighting recent squared
returns more heavily, with weights decaying geometrically into the past:

    sigma_t**2 = (1 - lambda) * sum_{k>=0} lambda**k * r_{t-1-k}**2

The decay ``lambda`` controls the memory: RiskMetrics' daily default is
**0.94**, meaning yesterday gets weight 0.06, the day before 0.0564, and so on.
A large recent move therefore pushes the VaR up immediately, and it relaxes back
as calm returns. Equivalently, the variance follows the one-line recursion

    sigma_t**2 = lambda * sigma_{t-1}**2 + (1 - lambda) * r_{t-1}**2,

which is what we actually iterate. The return is then taken Normal with this
conditional volatility, so VaR and ES are the Gaussian closed forms evaluated at
the *current* (end-of-sample) sigma -- the volatility that matters for tomorrow.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel
from varlib.models.parametric_brownian import normal_pdf, normal_quantile


class EwmaVar(VarModel):
    """EWMA / RiskMetrics (volatility-clustering) parametric VaR and ES."""

    method_name = "ewma"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        lambda_: float = 0.94,
        mu: float | None = None,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        if not 0.0 < lambda_ < 1.0:
            raise ValueError(f"lambda must be in (0, 1), got {lambda_}")
        self.lambda_ = float(lambda_)
        self.mu_override = mu

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        return ewma_var_es(
            returns,
            self.confidence,
            self.lambda_,
            self.mu_override,
            steps,
            horizon=self.horizon,
        )


def ewma_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    lambda_: float = 0.94,
    mu: float | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Compute EWMA VaR and ES at `horizon`, as positive loss fractions.

    The conditional one-day volatility is the exponentially weighted RMS of past
    returns (decay `lambda_`). Returns are taken Normal with that volatility, so
    -- exactly as in the Gaussian model -- the h-day mean is ``h*mu`` and the
    h-day standard deviation is ``sqrt(h)*sigma`` (the RiskMetrics square-root-of-
    time scaling, which is internally consistent with the i.i.d.-Normal-increment
    assumption EWMA makes about the *standardised* return). VaR and ES are then
    the Gaussian closed forms at that horizon volatility.

    Parameters
    ----------
    lambda_
        Decay factor in (0, 1). RiskMetrics' daily default is 0.94.
    mu
        Optional drift override. RiskMetrics assumes zero drift over short
        horizons; if omitted here we still use the sample mean so the model is
        consistent with the others, but it is typically tiny.
    """
    if steps is None:
        steps = {}

    returns = np.asarray(returns, dtype=float)
    horizon = int(horizon)
    n = returns.size
    if n < 2:
        raise ValueError("Need at least two returns for an EWMA volatility.")

    # Step 1: the drift (estimate or accept). Usually negligible at daily scale.
    if mu is None:
        mu = float(np.mean(returns))
        steps["mu_source"] = "estimated"
    else:
        mu = float(mu)
        steps["mu_source"] = "override"
    steps["mu"] = mu
    steps["lambda"] = float(lambda_)

    # Step 2: iterate the EWMA variance recursion across the window. We seed it
    # with the plain sample variance so the early estimate is reasonable, then let
    # the recursion pull it toward the recent regime. The squared deviations from
    # the mean are the inputs; the final value is the conditional variance for the
    # NEXT period -- the one we are forecasting.
    deviations = returns - mu
    sq = deviations * deviations
    sigma2 = float(np.var(returns, ddof=0))      # seed
    for k in range(n):
        sigma2 = lambda_ * sigma2 + (1.0 - lambda_) * sq[k]
    sigma = float(np.sqrt(sigma2))
    steps["ewma_variance"] = sigma2
    steps["sigma"] = sigma

    # For transparency, also record the effective weight on the most recent day
    # and the half-life of the decay (how many days until a weight halves).
    steps["weight_on_last_day"] = float(1.0 - lambda_)
    steps["half_life_days"] = float(np.log(0.5) / np.log(lambda_))

    # Step 3: project to the horizon (mean and variance add under i.i.d.).
    mu_h = horizon * mu
    sigma_h = float(np.sqrt(horizon)) * sigma
    if horizon > 1:
        steps["horizon"] = horizon
        steps["mu_horizon"] = mu_h
        steps["sigma_horizon"] = sigma_h

    # Step 4: Gaussian tail quantile, VaR and ES at the conditional volatility.
    tail_probability = 1.0 - confidence
    z = normal_quantile(tail_probability)
    steps["z_score"] = z

    return_at_quantile = mu_h + z * sigma_h
    steps["return_at_quantile"] = return_at_quantile
    var = float(-return_at_quantile)
    steps["var"] = var

    pdf_z = normal_pdf(z)
    es = float(-mu_h + sigma_h * pdf_z / tail_probability)
    steps["es"] = es

    return var, es


def ewma_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    lambda_: float = 0.94,
    mu: float | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> float:
    """Convenience wrapper returning only the EWMA VaR."""
    var, _ = ewma_var_es(returns, confidence, lambda_, mu, steps, horizon)
    return var

"""
EWMA / RiskMetrics VaR -- volatility that remembers recent days more.

This module implements the volatility-forecasting model of J.P. Morgan/Reuters,
*RiskMetrics(TM) -- Technical Document*, 4th edition, 1996 (the PDF sits next to
this file). All page numbers below are the **printed** page numbers of that
document. The relevant material is Chapter 5, "Estimation and forecast" (Peter
Zangari), with the VaR step coming from Chapter 4, Section 4.5.

The Gaussian model estimates one volatility from the whole window, so a calm day
a year ago counts exactly as much as yesterday's crash. Markets do not behave
that way: volatility *clusters* -- turbulent days follow turbulent days. The
exponentially weighted moving average (EWMA) fixes this by weighting recent
squared returns more heavily, with weights decaying geometrically into the past.

THE MODEL, EQUATION BY EQUATION (RiskMetrics 4th ed.)
-----------------------------------------------------
* Table 5.1, p.78 -- the EWMA volatility estimator, the weighted-sum form

      sigma**2 = (1 - lambda) * sum_{t=1..T} lambda**(t-1) * (r_t - r_bar)**2

  with decay factor ``lambda`` in (0, 1).

* Eq. [5.3], p.81 -- the *recursive* form, which is what we actually iterate
  (and what RiskMetrics uses for production forecasting). For the one-day-ahead
  variance forecast given data through day t:

      sigma**2_{t+1|t} = lambda * sigma**2_{t|t-1} + (1 - lambda) * r_t**2

  The document derives this "assuming an infinite amount of data are available"
  and "assuming again that the sample mean is zero" (p.81), so the input is the
  raw squared return r_t**2 -- NOT a deviation from the mean (see DEPARTURES).

* Eq. [5.4], p.81 -- the 1-day volatility forecast is sqrt of the above.

* p.80 -- the daily decay default: the document "arbitrarily choose[s]
  lambda = 0.94" for the worked example; Section 5.3.2 (pp.97-100) derives 0.94
  as the value that minimises forecast error across J.P. Morgan's data set.
  p.81 also notes this puts "a 6% weight to the most recent squared return".

* Eqs. [5.19]-[5.20], p.86 -- "square root of time" scaling. Because the EWMA
  forecast for every future day equals the next-day forecast (Eq. [5.18]), the
  T-day variance is T * sigma**2_{t+1}, hence sigma_T = sqrt(T) * sigma_{t+1}.

* Eqs. [4.41]-[4.44], p.69 (Chart 4.18) -- the VaR step. Standardise the return,
  r_tilde = (r_t - mu_t) / sigma_t ~ N(0, 1) [4.41], so

      Probability( r_t < mu_t - 1.65 * sigma_t ) = 5%        [4.43]

  i.e. the alpha-quantile of the return is mu + z_alpha * sigma with
  z_alpha = Phi^{-1}(alpha) (z = -1.65 at 95%, -2.33 at 99%, p.69). VaR is the
  negative of that quantile. Setting mu_t = 0 gives Eq. [4.44], which the
  document calls "the basis for short-term horizon VaR calculation".

WHERE THIS IMPLEMENTATION DEPARTS FROM THE LITERAL DOCUMENT
-----------------------------------------------------------
1. DRIFT mu. RiskMetrics sets the mean to zero -- emphatically and repeatedly:
   "keep matters simple by setting the sample mean ... to zero" (p.80),
   "assuming again that the sample mean is zero" (p.81). Its recursion [5.3]
   therefore feeds in raw r_t**2. We instead default ``mu`` to the sample mean
   and demean the returns before squaring, so that EWMA is consistent with the
   other parametric models in this library. At daily scale mu is tiny, so the
   numerical effect is negligible, but it is a genuine departure. Pass
   ``mu=0.0`` to recover the literal RiskMetrics model.

2. SEEDING. Eq. [5.3] is derived "assuming an infinite amount of data are
   available" (p.81) -- there is no initial condition in the document. With a
   finite window we must seed the recursion; we use the plain sample variance.
   Its influence decays as lambda**n and is gone within a few half-lives.

3. EXPECTED SHORTFALL. The 1996 document predates coherent risk measures and
   defines NO expected shortfall (ES / CVaR was formalised by Artzner et al.,
   1999, and Acerbi-Tasche, 2002). The ``es`` returned here is the standard
   Gaussian closed form ES = -mu_h + sigma_h * phi(z) / (1 - alpha), evaluated
   at the EWMA conditional volatility. It is a faithful extension of the same
   conditional-normal assumption, but it is not from RiskMetrics.

REPRODUCING THE DOCUMENT. Table 5.2 (p.81) works a 20-day USD/DEM example with
lambda = 0.94 and mean zero, reporting an exponentially weighted std of 0.333%.
That table uses the *truncated, non-renormalised* finite weighted sum (its
weights sum to ~0.71). The recursion we use here is the proper infinite-history
limit RiskMetrics actually recommends; the equally weighted column (0.393%)
reproduces exactly.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel
from varlib.models.parametric.brownian.brownian import normal_pdf, normal_quantile


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
    Compute EWMA VaR and ES at `horizon`

    Implements RiskMetrics(TM) -- Technical Document, 4th ed. (1996); page
    citations below are the document's printed pages. The conditional one-day
    volatility is the exponentially weighted RMS of past returns (Eq. [5.3],
    p.81). Returns are taken Normal with that volatility (Eq. [4.41], p.69), so
    -- exactly as in the Gaussian model -- the h-day mean is ``h*mu`` and the
    h-day standard deviation is ``sqrt(h)*sigma`` (the square-root-of-time
    scaling, Eqs. [5.19]-[5.20], p.86). VaR and ES are then the Gaussian closed
    forms at that horizon volatility (Eqs. [4.42]-[4.44], p.69).

    Parameters
    ----------
    lambda_
        Decay factor in (0, 1). RiskMetrics' daily default is 0.94 (p.80;
        derived in Section 5.3.2, pp.97-100).
    mu
        Optional drift override. RiskMetrics assumes zero drift over short
        horizons ("setting the sample mean ... to zero", p.80). DEPARTURE: if
        omitted we use the sample mean for consistency with the library's other
        models; pass ``mu=0.0`` for the literal RiskMetrics model. It is
        typically tiny at daily scale.
    """
    if steps is None:
        steps = {}

    returns = np.asarray(returns, dtype=float)
    horizon = int(horizon)
    n = returns.size
    if n < 2:
        raise ValueError("Need at least two returns for an EWMA volatility. But really, a lot more than two for a meaningful estimate.")

    # Step 1: Drift
    if mu is None:
        mu = float(np.mean(returns))
        steps["mu_source"] = "estimated"
    else:
        mu = float(mu)
        steps["mu_source"] = "override"
    steps["mu"] = mu
    steps["lambda"] = float(lambda_)

    # Step 2: iterate the EWMA variance recursion, Eq. [5.3], p.81:
    #     sigma**2_{t+1|t} = lambda * sigma**2_{t|t-1} + (1 - lambda) * r_t**2
    # The document derives this assuming an infinite history and a zero mean. 
    # Two DEPARTURES here, both noted in the module docstring: (a) we feed squared 
    # deviations from the mean rather than raw squares, and (b) we must seed the recursion 
    # -- the document, assuming infinite data, never does. We seed with the plain 
    # sample variance so the early estimate is reasonable;
    deviations = returns - mu
    sq = deviations * deviations
    sigma2 = float(np.var(returns, ddof=0))      # seed (not in the document)
    for k in range(n):
        sigma2 = lambda_ * sigma2 + (1.0 - lambda_) * sq[k]
    sigma = float(np.sqrt(sigma2))               # 1-day forecast, Eq. [5.4], p.81
    steps["ewma_variance"] = sigma2
    steps["sigma"] = sigma

    # For transparency, also record the effective weight on the most recent day
    # (1 - lambda; "a 6% weight to the most recent squared return", p.81) and the
    # half-life of the decay (how many days until a weight halves).
    steps["weight_on_last_day"] = float(1.0 - lambda_)
    steps["half_life_days"] = float(np.log(0.5) / np.log(lambda_))

    # Step 3: project to the horizon via square-root-of-time, Eqs. [5.19]-[5.20],
    # p.86. Because the EWMA forecast is flat across future days (Eq. [5.18]),
    # the T-day variance is T * sigma**2 and mean and variance add under i.i.d.
    mu_h = horizon * mu
    sigma_h = float(np.sqrt(horizon)) * sigma
    if horizon > 1:
        steps["horizon"] = horizon
        steps["mu_horizon"] = mu_h
        steps["sigma_horizon"] = sigma_h

    # Step 4: Gaussian tail quantile, VaR and ES at the conditional volatility.
    # Standardise (Eq. [4.41], p.69), so the alpha-quantile of the return is
    # mu_h + z * sigma_h with z = Phi^{-1}(1 - confidence) (z = -1.65 at 95%,
    # -2.33 at 99%, p.69). VaR is the negative of that quantile, Eqs. [4.43]-
    # [4.44], p.69.
    tail_probability = 1.0 - confidence
    z = normal_quantile(tail_probability)
    steps["z_score"] = z

    return_at_quantile = mu_h + z * sigma_h
    steps["return_at_quantile"] = return_at_quantile
    var = float(-return_at_quantile)
    steps["var"] = var

    # ES: NOT in the 1996 document (it predates coherent risk measures). This is
    # the Gaussian closed form ES = -mu_h + sigma_h * phi(z) / (1 - alpha)
    # evaluated at the EWMA volatility -- a faithful extension of the same
    # conditional-normal assumption (Acerbi-Tasche, 2002), not RiskMetrics.
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

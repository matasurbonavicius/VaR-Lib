"""
Historical VaR -- the empirical quantile of past losses.

It makes no assumption about the shape of the return distribution: it simply asks "over the historical window,
how bad was the loss that we exceeded only (1 - confidence) of the time?".

The whole method is two lines of arithmetic:

    VaR = the loss at the confidence-level quantile
    ES  = the average of the losses at or beyond that VaR

The obvious problem: It can't think ahead of the historical window. If the next 1000 days are 
all worse than the worst day in the historical window, then the VaR and ES will be too low and vice versa. 
This is a problem for any non-parametric method, really but it is particularly acute for historical VaR. Therefore:

< Historical VaR assumes that future is like the past. >

Additionally, this is such a simple model that no paper is written about it. For a short description see the reference.

Reference: JPMorgan_RiskMetrics_TechnicalDocument.pdf, page 27
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import (
    VarModel,
    cumulative_returns,
    var_es_from_returns,
)


class HistoricalVar(VarModel):
    """
    Empirical-quantile (historical simulation) VaR and ES.

    The exact sequence to derive historical VaR and ES is:
    1. Take returns.
    2. Sum h-day returns → for horizon 30, a rolling 30-day window sum.
    3. Sort the losses.
    4. 95th percentile = VaR; average beyond it = ES.
    """

    method_name = "historical"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        overlapping: bool = True,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        self.overlapping = bool(overlapping)

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        # For a multi-day horizon the VaR is the quantile of the h-day loss
        # distribution, so we read it off h-day cumulative returns. For horizon 1 
        # this is just the returns themselves, so the one-day path is unchanged.

        # FYI: usually historical simulation is calculated on overlapping returns (for multi-day horizons)
        horizon_returns = cumulative_returns(
            returns, self.horizon, self.overlapping
        )
        
        steps["horizon"] = self.horizon
        if self.horizon > 1:
            steps["overlapping"] = self.overlapping
            steps["horizon_returns"] = horizon_returns

        # VaR is the loss quantile at the confidence level; ES is the average
        # loss at or beyond it (see var_es_from_returns for the definition).
        steps["losses"] = -np.asarray(horizon_returns, dtype=float)
        var, es = var_es_from_returns(horizon_returns, self.confidence)
        steps["var"] = var
        steps["es"] = es

        return var, es

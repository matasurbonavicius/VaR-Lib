"""
Historical VaR -- the empirical quantile of past losses.

This is the workhorse of bank risk management. It makes no assumption about the
shape of the return distribution: it simply asks "over the historical window,
how bad was the loss that we exceeded only (1 - confidence) of the time?".

Worked example, 99% confidence
------------------------------
If we have 1000 daily returns, then 1% of them is 10 observations. The 99% VaR
is the loss level such that only those 10 worst days were worse. We read it
straight off the sorted list of losses.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel, overlapping_cumulative_returns


class HistoricalVar(VarModel):
    """Empirical-quantile (historical simulation) VaR and ES."""

    method_name = "historical"

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        # For a multi-day horizon the VaR is the quantile of the h-day loss
        # distribution, so we read it off overlapping h-day cumulative returns
        # rather than scaling the one-day number. For horizon 1 this is just the
        # returns themselves, so the one-day path is unchanged.
        horizon_returns = overlapping_cumulative_returns(returns, self.horizon)
        if self.horizon > 1:
            steps["horizon"] = self.horizon
            steps["horizon_returns"] = horizon_returns
        var = historical_var(horizon_returns, self.confidence, steps)
        es = historical_es(horizon_returns, self.confidence, var, steps)
        return var, es


def historical_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    steps: dict[str, Any] | None = None,
) -> float:
    """
    Compute one-period Historical VaR as a positive loss fraction.

    Parameters
    ----------
    returns
        1-D array of per-period returns.
    confidence
        Confidence level, e.g. 0.99.
    steps
        Optional trace dictionary; every intermediate is recorded into it.
    """
    if steps is None:
        steps = {}

    # Step 1: a VaR is about losses, so flip the sign of returns.
    # A return of -0.02 (a 2% drop) becomes a loss of +0.02.
    losses = -np.asarray(returns, dtype=float)
    steps["losses"] = losses

    # Step 2: sort the losses from smallest to largest.
    sorted_losses = np.sort(losses)
    steps["sorted_losses"] = sorted_losses

    # Step 3: the VaR is the loss quantile at the confidence level.
    # We use linear interpolation between order statistics, which is the
    # standard, well-defined empirical-quantile estimator.
    var = float(np.quantile(sorted_losses, confidence, method="linear"))
    steps["var"] = var

    return var


def historical_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    var: float | None = None,
    steps: dict[str, Any] | None = None,
) -> float:
    """
    Compute one-period Historical Expected Shortfall as a positive loss fraction.

    ES is the average loss on the days whose loss was at least as bad as the
    VaR. It is the "expected loss given a breach".

    Parameters
    ----------
    var
        The VaR threshold to average beyond. If omitted it is computed from the
        same returns at the same confidence.
    """
    if steps is None:
        steps = {}

    # Step 1: losses, and the VaR threshold to look beyond.
    losses = -np.asarray(returns, dtype=float)
    if var is None:
        var = historical_var(returns, confidence)
    steps["es_threshold"] = float(var)

    # Step 2: pick out the losses in the tail (at or beyond the VaR).
    tail_losses = losses[losses >= var]
    steps["tail_losses"] = tail_losses

    # Step 3: ES is the average of those tail losses. If nothing reaches the
    # threshold (tiny samples), fall back to the VaR itself.
    if tail_losses.size == 0:
        es = float(var)
    else:
        es = float(np.mean(tail_losses))
    steps["es"] = es

    return es

"""
Breach counting and the Basel traffic-light test

Regulators do not run a likelihood-ratio test; they count breaches over the last
250 *trading days* and place the model in a zone:

  * Green  (0-4 breaches at 99%): the model is fine.
  * Yellow (5-9 breaches):        watch it; capital add-on increases with count.
  * Red    (10+ breaches):        the model is rejected.

The 250 is a fixed regulatory constant -- one year of daily observations -- and
the zone boundaries below are derived for exactly that count. It is a *daily*
test: each observation is one trading day's forecast-vs-realised, so a series
sampled at any finer frequency (intraday bars, minutely, ...) must be aggregated
to daily returns before this test is meaningful. Feeding it 1000+ observations,
or intraday bars, silently widens the "green" band and stops being the Basel
test.

Reference: https://www.bis.org/publ/bcbs22.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import binom


@dataclass
class BreachSummary:
    """Where the breaches are and how many there are."""

    n_observations: int
    n_breaches: int
    breach_rate: float
    breach_indices: list[int]
    steps: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrafficLightResult:
    """Basel traffic-light zone for a breach count."""

    n_observations: int
    n_breaches: int
    zone: str                  # "green", "yellow", or "red"
    cumulative_probability: float
    green_max: int
    red_min: int
    steps: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"TrafficLightResult(breaches={self.n_breaches}/{self.n_observations}, "
            f"zone={self.zone.upper()})"
        )


def count_breaches(realised_losses: np.ndarray, var_forecasts: np.ndarray) -> BreachSummary:
    """
    Flag every day where the realised loss exceeded the VaR forecast.

    Parameters
    ----------
    realised_losses
        1-D array of realised losses (positive = a loss), one per day.
    var_forecasts
        1-D array of VaR forecasts (positive numbers), aligned day-for-day.
    """
    steps: dict[str, Any] = {}

    realised_losses = np.asarray(realised_losses, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    if realised_losses.shape != var_forecasts.shape:
        raise ValueError("realised_losses and var_forecasts must be the same length.")

    n = int(realised_losses.size)
    if n == 0:
        raise ValueError("Need at least one observation.")

    # Step 1: a breach is a day whose realised loss exceeded the forecast VaR.
    is_breach = realised_losses > var_forecasts
    steps["is_breach"] = is_breach

    # Step 2: count them and note where they happened.
    breach_indices = [int(i) for i in np.flatnonzero(is_breach)]
    n_breaches = len(breach_indices)
    breach_rate = n_breaches / n
    steps["n_breaches"] = n_breaches
    steps["breach_rate"] = breach_rate
    steps["breach_indices"] = breach_indices

    return BreachSummary(
        n_observations=n,
        n_breaches=n_breaches,
        breach_rate=breach_rate,
        breach_indices=breach_indices,
        steps=steps,
    )


def breach_count(realised_losses: np.ndarray, var_forecasts: np.ndarray) -> BreachSummary:
    """Alias for `count_breaches`, kept for the name used in the README."""
    return count_breaches(realised_losses, var_forecasts)


BASEL_WINDOW = 250  # regulatory constant: one year of daily observations


def basel_traffic_light(
    n_breaches: int,
    n_observations: int = BASEL_WINDOW,
    confidence: float = 0.99,
) -> TrafficLightResult:
    """
    Assign a Basel traffic-light zone to a breach count.

    The zone boundaries are derived from the cumulative binomial distribution of
    the breach count under a correctly specified model, exactly as in the Basel
    framework.

    ``n_observations`` should be the Basel window of 250 trading days (the
    default). It is exposed only for testing and for confidence levels that
    genuinely use a different regulatory window -- passing the full length of a
    multi-year backtest here is a mistake: it widens the "green" band far beyond
    0-4 breaches and stops being the Basel test. To zone a long backtest, use
    :func:`basel_traffic_light_trailing`, which counts breaches over just the
    most recent 250 days.
    """
    steps: dict[str, Any] = {}

    if n_breaches < 0 or n_observations < 1 or n_breaches > n_observations:
        raise ValueError("Invalid breach count or observation count.")

    expected_rate = 1.0 - confidence
    steps["expected_rate"] = expected_rate
    steps["n_observations"] = n_observations

    # Step 1: cumulative binomial probability P(X <= k) for each possible count
    # k = 0..n, under a correctly specified model where X ~ Binomial(n, rate).
    counts = np.arange(n_observations + 1)
    cumulative = binom.cdf(counts, n_observations, expected_rate)
    steps["cumulative_at_count"] = float(cumulative[n_breaches])

    # Step 2: locate the zone boundaries from the cumulative distribution.
    # Yellow begins at the first count whose cumulative probability reaches 95%;
    # green is therefore everything below that. Red begins at the first count
    # whose cumulative probability reaches 99.99%.
    yellow_start = int(np.searchsorted(cumulative, 0.95, side="left"))
    green_max = yellow_start - 1
    red_min = int(np.searchsorted(cumulative, 0.9999, side="left"))
    steps["green_max"] = green_max
    steps["red_min"] = red_min

    # Step 3: place this count in its zone.
    if n_breaches <= green_max:
        zone = "green"
    elif n_breaches < red_min:
        zone = "yellow"
    else:
        zone = "red"
    steps["zone"] = zone

    return TrafficLightResult(
        n_observations=n_observations,
        n_breaches=n_breaches,
        zone=zone,
        cumulative_probability=float(cumulative[n_breaches]),
        green_max=green_max,
        red_min=red_min,
        steps=steps,
    )


def basel_traffic_light_trailing(
    realised_losses: np.ndarray,
    var_forecasts: np.ndarray,
    confidence: float = 0.99,
    window: int = BASEL_WINDOW,
) -> TrafficLightResult:
    """
    Basel traffic-light zone over the *most recent* ``window`` trading days.

    The Basel test is a rolling 250-day check, not a full-history one: it counts
    breaches over the last ``window`` observations and zones that count against
    the fixed 250-day boundaries. Feeding a whole multi-year backtest to
    :func:`basel_traffic_light` instead widens the green band and is not the
    Basel test.

    The inputs are the aligned daily ``(realised_losses, var_forecasts)`` from a
    roll (see :func:`varlib.backtest.rolling_backtest`); each observation is one
    trading day. If fewer than ``window`` observations are available, all of them
    are used and ``n_observations`` reflects the shorter count.
    """
    realised_losses = np.asarray(realised_losses, dtype=float)
    var_forecasts = np.asarray(var_forecasts, dtype=float)
    if realised_losses.shape != var_forecasts.shape:
        raise ValueError("realised_losses and var_forecasts must be the same length.")
    if realised_losses.size == 0:
        raise ValueError("Need at least one observation.")

    tail = min(window, realised_losses.size)
    recent = count_breaches(realised_losses[-tail:], var_forecasts[-tail:])
    return basel_traffic_light(recent.n_breaches, tail, confidence)

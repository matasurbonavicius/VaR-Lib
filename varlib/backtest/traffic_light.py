"""
Breach counting and the Basel traffic-light test -- the supervisory view.

Regulators do not run a likelihood-ratio test; they count breaches over the last
250 trading days and place the model in a zone:

  * Green  (0-4 breaches at 99%): the model is fine.
  * Yellow (5-9 breaches):        watch it; capital add-on increases with count.
  * Red    (10+ breaches):        the model is rejected.

The boundaries come from the cumulative binomial: green covers up to the 95th
percentile of the breach count, red starts where the cumulative probability
exceeds 99.99%. We compute those probabilities directly so the zoning is
transparent rather than hard-coded magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


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


def basel_traffic_light(
    n_breaches: int,
    n_observations: int = 250,
    confidence: float = 0.99,
) -> TrafficLightResult:
    """
    Assign a Basel traffic-light zone to a breach count.

    The zone boundaries are derived from the cumulative binomial distribution of
    the breach count under a correctly specified model, exactly as in the Basel
    framework.
    """
    steps: dict[str, Any] = {}

    if n_breaches < 0 or n_observations < 1 or n_breaches > n_observations:
        raise ValueError("Invalid breach count or observation count.")

    expected_rate = 1.0 - confidence
    steps["expected_rate"] = expected_rate
    steps["n_observations"] = n_observations

    # Step 1: cumulative binomial probability P(X <= k) for each possible count.
    cumulative = _binomial_cdf_table(n_observations, expected_rate)
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


def _binomial_cdf_table(n: int, p: float) -> np.ndarray:
    """
    Return P(X <= k) for k = 0..n where X ~ Binomial(n, p).

    Built by recurrence on the probability mass function, which avoids large
    factorials:  pmf(k+1) = pmf(k) * (n - k) / (k + 1) * p / (1 - p).
    """
    pmf = np.empty(n + 1, dtype=float)
    pmf[0] = (1.0 - p) ** n
    ratio = p / (1.0 - p)
    for k in range(n):
        pmf[k + 1] = pmf[k] * (n - k) / (k + 1) * ratio
    return np.cumsum(pmf)

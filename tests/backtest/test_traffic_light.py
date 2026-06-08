"""Tests for breach counting and the Basel traffic-light zoning."""

import numpy as np
import pytest

from varlib.backtest import basel_traffic_light, count_breaches


def test_count_breaches_flags_exceedances():
    realised = np.array([0.01, 0.05, 0.02, 0.08])
    forecast = np.array([0.03, 0.03, 0.03, 0.03])
    summary = count_breaches(realised, forecast)
    # Losses 0.05 and 0.08 exceed the 0.03 forecast.
    assert summary.n_breaches == 2
    assert summary.breach_indices == [1, 3]


def test_count_breaches_rate():
    realised = np.zeros(100)
    realised[:3] = 1.0
    forecast = np.full(100, 0.5)
    summary = count_breaches(realised, forecast)
    assert summary.breach_rate == pytest.approx(0.03)


def test_basel_zones_at_250_days():
    # Classic Basel 99% / 250-day zoning: green 0-4, yellow 5-9, red 10+.
    assert basel_traffic_light(4, 250, 0.99).zone == "green"
    assert basel_traffic_light(5, 250, 0.99).zone == "yellow"
    assert basel_traffic_light(9, 250, 0.99).zone == "yellow"
    assert basel_traffic_light(10, 250, 0.99).zone == "red"


def test_zone_boundaries_are_derived_not_hardcoded():
    result = basel_traffic_light(0, 250, 0.99)
    assert result.green_max == 4
    assert result.red_min == 10


def test_rejects_impossible_counts():
    with pytest.raises(ValueError):
        basel_traffic_light(300, 250, 0.99)


def test_mismatched_lengths_rejected():
    with pytest.raises(ValueError):
        count_breaches(np.array([0.1, 0.2]), np.array([0.1]))

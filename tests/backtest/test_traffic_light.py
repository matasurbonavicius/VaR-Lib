"""Tests for breach counting and the Basel traffic-light zoning."""

import numpy as np
import pytest

from varlib.backtest import (
    basel_traffic_light,
    basel_traffic_light_trailing,
    count_breaches,
)


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


def test_trailing_zones_only_the_last_250():
    # A long backtest: 10 breaches spread over the first 750 days (an old crisis)
    # and only 2 in the most recent 250. Full-history zoning would swallow all 12
    # into a wide "green" band; the trailing test must see just the recent 2.
    n = 1000
    forecast = np.full(n, 0.03)
    realised = np.full(n, 0.01)          # calm by default (no breach)
    realised[:10] = 0.05                 # 10 old breaches, outside the last 250
    realised[[900, 950]] = 0.05          # 2 recent breaches, inside the last 250

    result = basel_traffic_light_trailing(realised, forecast, 0.99)
    assert result.n_observations == 250
    assert result.n_breaches == 2        # only the recent window is counted
    assert result.zone == "green"        # 2 <= 4


def test_trailing_flags_a_recent_breach_cluster_as_red():
    # 12 breaches all inside the most recent 250 days -> red (10+).
    n = 1000
    forecast = np.full(n, 0.03)
    realised = np.full(n, 0.01)
    realised[-12:] = 0.05
    result = basel_traffic_light_trailing(realised, forecast, 0.99)
    assert result.n_observations == 250
    assert result.n_breaches == 12
    assert result.zone == "red"


def test_trailing_uses_all_data_when_shorter_than_window():
    # Fewer than 250 observations: zone against what we have, not a padded 250.
    forecast = np.full(100, 0.03)
    realised = np.full(100, 0.01)
    result = basel_traffic_light_trailing(realised, forecast, 0.99)
    assert result.n_observations == 100

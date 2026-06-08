"""Tests for the Basel traffic-light chart."""

import pytest

from varlib.backtest import basel_traffic_light
from varlib.plotting import traffic_light_chart


def test_draws_three_zones_and_a_marker():
    ax = traffic_light_chart(n_breaches=3, n_observations=250, confidence=0.99)
    # Three axvspan patches (the zones) plus a single marker line at the count
    # (no dot -- the line alone marks the position).
    assert len(ax.patches) >= 3
    assert len(ax.lines) >= 1
    assert ax.lines[0].get_xdata()[0] == 3
    assert "GREEN" in ax.texts[0].get_text().upper()


def test_accepts_a_precomputed_result():
    result = basel_traffic_light(7, 250, 0.99)  # yellow zone (green<=4, red>=10)
    ax = traffic_light_chart(result=result)
    assert "YELLOW" in ax.texts[0].get_text().upper()


def test_red_zone_count():
    ax = traffic_light_chart(n_breaches=15, n_observations=250, confidence=0.99)
    assert "RED" in ax.texts[0].get_text().upper()


def test_requires_some_input():
    with pytest.raises(ValueError):
        traffic_light_chart()

"""Tests for the breach timeline chart."""

import numpy as np

from varlib.plotting import breach_timeline


def test_draws_one_tick_per_breach():
    flags = np.zeros(100)
    flags[[10, 40, 90]] = 1
    ax = breach_timeline(flags)
    # The vlines call adds a single LineCollection holding all the ticks.
    assert len(ax.collections) >= 1
    assert "3 breaches" in ax.get_title(loc="left")


def test_anchors_to_full_date_range():
    import pandas as pd
    flags = np.zeros(50)
    flags[25] = 1
    dates = pd.date_range("2020-01-01", periods=50, freq="B")
    ax = breach_timeline(flags, dates=dates)
    left, right = ax.get_xlim()
    # The axis should span the whole period, not just the single breach.
    assert right - left > 30  # comfortably more than one day in date units


def test_handles_no_breaches():
    ax = breach_timeline(np.zeros(20))
    assert "0 breaches" in ax.get_title(loc="left")

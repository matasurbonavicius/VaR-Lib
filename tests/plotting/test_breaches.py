"""Tests for the breaches chart."""

import numpy as np
import pytest

from varlib.plotting import breaches_chart


def test_returns_axes_with_expected_artists(backtest_data):
    losses, forecasts = backtest_data
    ax = breaches_chart(losses, forecasts, confidence=0.99)
    # The VaR line, the loss bars, and the breach scatter should all be present.
    assert ax.get_title(loc="left")  # styled title set (left-aligned)
    assert len(ax.lines) >= 1          # VaR forecast line
    assert len(ax.collections) >= 1    # breach scatter
    assert ax.get_legend() is not None


def test_breach_count_matches_data(backtest_data):
    losses, forecasts = backtest_data
    ax = breaches_chart(losses, forecasts)
    expected = int((losses > forecasts).sum())
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any(f"Breaches ({expected})" == lbl for lbl in labels)


def test_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        breaches_chart(np.zeros(10), np.zeros(9))


def test_accepts_dates(backtest_data):
    losses, forecasts = backtest_data
    import pandas as pd
    dates = pd.date_range("2020-01-01", periods=losses.size, freq="B")
    ax = breaches_chart(losses, forecasts, dates=dates)
    assert ax.get_xlabel() == "Date"

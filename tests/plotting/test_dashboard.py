"""Tests for the combined backtest dashboard."""

import numpy as np

from varlib.plotting import backtest_dashboard


def test_dashboard_has_four_panels(backtest_data):
    losses, forecasts = backtest_data
    fig = backtest_dashboard(losses, forecasts, confidence=0.99)
    assert len(fig.axes) == 4


def test_dashboard_es_line_not_inside_var(backtest_data):
    # The dashboard derives ES from the same sample as VaR, so on the
    # distribution panel the ES line must not sit inside the VaR line.
    losses, forecasts = backtest_data
    fig = backtest_dashboard(losses, forecasts, confidence=0.99)
    dist_ax = fig.axes[-1]  # distribution panel is added last
    var_x = dist_ax.lines[0].get_xdata()[0]
    es_x = dist_ax.lines[1].get_xdata()[0]
    assert es_x <= var_x


def test_dashboard_accepts_dates(backtest_data):
    import pandas as pd
    losses, forecasts = backtest_data
    dates = pd.date_range("2020-01-01", periods=losses.size, freq="B")
    fig = backtest_dashboard(losses, forecasts, dates=dates, confidence=0.99)
    assert len(fig.axes) == 4

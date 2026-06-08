"""Tests for the examples' ``roll_backtest`` wrapper.

``roll_backtest`` is now a thin wrapper over the library's
``varlib.backtest.rolling_backtest`` (see ``tests/test_rolling.py`` for the
helper itself). These tests pin down the wrapper's behaviour through the example
entry point -- the same alignment guarantee, reached the way the examples reach
it -- so the example scripts stay covered after the move into the package.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# The example wrapper lives alongside the other example helpers.
EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples")
sys.path.insert(0, EXAMPLES)

from _common import roll_backtest  # noqa: E402
from varlib import HistoricalVar  # noqa: E402


def _price_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


def test_horizon_one_realised_is_single_day_loss():
    prices = _price_series()
    model = HistoricalVar(0.99, horizon=1)
    losses, forecasts, dates = roll_backtest(prices, model, window=250)
    # With horizon 1, realised loss is exactly the negative one-day log return.
    from varlib._returns import to_returns

    rets = to_returns(prices.to_numpy())
    assert losses[0] == pytest.approx(-rets[250])
    assert len(losses) == len(forecasts) == len(dates)


def test_overlap_realised_is_cumulative_h_day_loss():
    prices = _price_series()
    horizon = 10
    model = HistoricalVar(0.99, horizon=horizon)
    losses, forecasts, dates = roll_backtest(prices, model, window=250, overlap=True)

    from varlib._returns import to_returns

    rets = to_returns(prices.to_numpy())
    # First realised loss is the negative sum of the next h daily returns.
    assert losses[0] == pytest.approx(-rets[250:250 + horizon].sum())
    # The date label is the END of the h-day window.
    assert dates[0] == prices.index[250 + horizon]


def test_overlap_vs_non_overlap_counts():
    prices = _price_series(n=600)
    horizon = 10
    window = 250
    model = HistoricalVar(0.99, horizon=horizon)

    n_overlap = len(roll_backtest(prices, model, window=window, overlap=True)[0])
    n_non = len(roll_backtest(prices, model, window=window, overlap=False)[0])

    n_returns = len(prices) - 1
    # Overlapping advances by 1; non-overlapping by `horizon`.
    expected_overlap = n_returns - horizon + 1 - window
    assert n_overlap == expected_overlap
    # Non-overlapping uses roughly 1/horizon as many points.
    assert n_non == pytest.approx(n_overlap / horizon, abs=2)
    assert n_non < n_overlap


def test_non_overlap_windows_do_not_overlap():
    # Consecutive non-overlapping realised losses come from disjoint day ranges,
    # so their end-dates are `horizon` business days apart.
    prices = _price_series()
    horizon = 5
    model = HistoricalVar(0.99, horizon=horizon)
    _, _, dates = roll_backtest(prices, model, window=250, overlap=False)
    gaps = np.diff([d.value for d in pd.to_datetime(dates)])
    # All gaps equal (each is exactly `horizon` business days).
    assert len(set(gaps)) == 1

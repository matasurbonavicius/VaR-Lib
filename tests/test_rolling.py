"""Tests for the library rolling-VaR helpers (varlib.backtest.rolling)."""

import numpy as np
import pandas as pd
import pytest

from varlib import HistoricalVar, rolling_backtest, rolling_var
from varlib._returns import to_returns


def _prices(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


# -- rolling_var: series in, series of VaR out ------------------------------


def test_rolling_var_produces_one_forecast_per_step():
    prices = _prices()
    model = HistoricalVar(0.99, horizon=1)
    forecasts = rolling_var(model, prices=prices, window=250)
    rets = to_returns(prices.to_numpy())
    # horizon 1: forecasts run from t=window .. last return inclusive.
    assert len(forecasts) == rets.size - 250
    assert np.all(forecasts > 0)


def test_rolling_var_matches_a_manual_fit():
    prices = _prices()
    model = HistoricalVar(0.99, horizon=1)
    rets = to_returns(prices.to_numpy())
    forecasts = rolling_var(model, returns=rets, window=250)
    # The first forecast must equal fitting the model on the first window.
    manual = HistoricalVar(0.99).run(returns=rets[:250]).value
    assert forecasts[0] == pytest.approx(manual)


def test_rolling_var_can_collect_es():
    prices = _prices()
    model = HistoricalVar(0.99)
    var_series = rolling_var(model, prices=prices, window=250, field="value")
    es_series = rolling_var(model, prices=prices, window=250, field="expected_shortfall")
    assert len(var_series) == len(es_series)
    # ES >= VaR at every step.
    assert np.all(es_series >= var_series - 1e-12)


def test_rolling_var_step_gives_fewer_points():
    prices = _prices()
    model = HistoricalVar(0.99, horizon=5)
    every_day = rolling_var(model, prices=prices, window=250, step=1)
    every_five = rolling_var(model, prices=prices, window=250, step=5)
    assert len(every_five) == pytest.approx(len(every_day) / 5, abs=2)
    assert len(every_five) < len(every_day)


def test_rolling_var_rejects_bad_field():
    with pytest.raises(ValueError):
        rolling_var(HistoricalVar(0.99), returns=np.zeros(300), window=250, field="bogus")


def test_rolling_var_requires_one_input():
    with pytest.raises(ValueError):
        rolling_var(HistoricalVar(0.99), window=10)  # neither returns nor prices


def test_rolling_var_window_too_large():
    with pytest.raises(ValueError):
        rolling_var(HistoricalVar(0.99), returns=np.zeros(100), window=250)


# -- rolling_backtest: aligned (losses, forecasts, dates) -------------------


def test_backtest_horizon_one_realised_is_single_day_loss():
    prices = _prices()
    model = HistoricalVar(0.99, horizon=1)
    losses, forecasts, dates = rolling_backtest(model, prices=prices, window=250)
    rets = to_returns(prices.to_numpy())
    assert losses[0] == pytest.approx(-rets[250])
    assert len(losses) == len(forecasts) == len(dates)


def test_backtest_overlap_realised_is_cumulative_h_day_loss():
    prices = _prices()
    horizon = 10
    model = HistoricalVar(0.99, horizon=horizon)
    losses, forecasts, dates = rolling_backtest(model, prices=prices, window=250, overlap=True)
    rets = to_returns(prices.to_numpy())
    assert losses[0] == pytest.approx(-rets[250:250 + horizon].sum())
    # The date label is the END of the h-day window, taken from the price index.
    assert dates[0] == prices.index[250 + horizon]


def test_backtest_dates_come_from_series_index():
    prices = _prices()
    model = HistoricalVar(0.99, horizon=1)
    _, _, dates = rolling_backtest(model, prices=prices, window=250)
    # Every date label is a real index entry, in order.
    assert all(d in set(prices.index) for d in dates)
    assert list(dates) == sorted(dates)


def test_backtest_integer_axis_when_no_index():
    # Plain numpy returns -> integer step labels, no pandas needed for the maths.
    rets = np.random.default_rng(1).normal(0, 0.01, 400)
    model = HistoricalVar(0.99, horizon=1)
    losses, forecasts, dates = rolling_backtest(model, returns=rets, window=250)
    assert dates[0] == 251  # t + horizon = 250 + 1
    assert all(isinstance(d, (int, np.integer)) for d in dates)


def test_backtest_non_overlap_is_sparser():
    prices = _prices(n=600)
    model = HistoricalVar(0.99, horizon=10)
    n_overlap = len(rolling_backtest(model, prices=prices, window=250, overlap=True)[0])
    n_non = len(rolling_backtest(model, prices=prices, window=250, overlap=False)[0])
    assert n_non < n_overlap
    assert n_non == pytest.approx(n_overlap / 10, abs=2)


def test_backtest_non_overlap_windows_are_disjoint():
    # Consecutive non-overlapping realised losses come from disjoint day ranges,
    # so their end-dates are exactly `horizon` business days apart -- one gap size.
    prices = _prices()
    model = HistoricalVar(0.99, horizon=5)
    _, _, dates = rolling_backtest(model, prices=prices, window=250, overlap=False)
    gaps = np.diff([d.value for d in pd.to_datetime(dates)])
    assert len(set(gaps)) == 1


def test_backtest_feeds_a_real_backtest():
    # End-to-end: the triple drops straight into the breach counter / Kupiec.
    from varlib.backtest import count_breaches, kupiec_pof_test

    prices = _prices()
    model = HistoricalVar(0.99)
    losses, forecasts, _ = rolling_backtest(model, prices=prices, window=250)
    summary = count_breaches(losses, forecasts)
    result = kupiec_pof_test(np.array(summary.steps["is_breach"], dtype=float), 0.99)
    assert 0.0 <= result.p_value <= 1.0

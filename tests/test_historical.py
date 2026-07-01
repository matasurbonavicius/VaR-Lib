"""Tests for Historical VaR."""

import numpy as np
import pytest

from varlib import HistoricalVar, historical_var, historical_es
from varlib.base import VarResult


def test_var_is_known_quantile_of_uniform_losses():
    # Returns from -0.01 .. -0.000 (so losses 0.000 .. 0.01, uniform-ish).
    returns = np.linspace(-0.01, 0.0, 101)
    var = historical_var(returns, confidence=0.99)
    # The 99% loss quantile of losses in [0, 0.01] should be near 0.0099.
    assert var == pytest.approx(0.0099, abs=1e-4)


def test_higher_confidence_gives_higher_var():
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.02, 5000)
    v95 = historical_var(returns, 0.95)
    v99 = historical_var(returns, 0.99)
    assert v99 > v95


def test_steps_are_traced():
    returns = np.array([-0.03, -0.01, 0.0, 0.01, 0.02])
    steps = {}
    historical_var(returns, 0.95, steps)
    assert set(steps) >= {"losses", "sorted_losses", "var"}
    # Losses are the negated returns.
    assert np.allclose(steps["losses"], -returns)
    # Sorted losses are non-decreasing.
    assert np.all(np.diff(steps["sorted_losses"]) >= 0)


def test_model_returns_var_result_with_full_trace():
    returns = np.random.default_rng(0).normal(0, 0.01, 1000)
    result = HistoricalVar(confidence=0.99).run(returns=returns)
    assert isinstance(result, VarResult)
    assert result.method == "historical"
    assert result.value > 0
    assert "var" in result.steps
    assert "n_observations" in result.steps


def test_accepts_prices_and_returns_equivalently():
    prices = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 100.0])
    returns = np.diff(np.log(prices))
    from_prices = HistoricalVar(0.95).run(returns).value
    from_returns = HistoricalVar(0.95).run(returns=returns).value
    assert from_prices == pytest.approx(from_returns)


def test_horizon_var_is_quantile_of_cumulative_returns():
    # The h-day VaR is computed directly: it is the historical quantile of the
    # OVERLAPPING h-day cumulative returns, not the one-day VaR scaled by sqrt(h).
    returns = np.random.default_rng(2).normal(0, 0.01, 1000)
    ten_day = HistoricalVar(0.99, horizon=10).run(returns=returns).value

    # Reconstruct the 10-day cumulative returns (log returns add) and take the
    # same quantile directly.
    from varlib.base import overlapping_cumulative_returns

    cum = overlapping_cumulative_returns(returns, 10)
    expected = float(np.quantile(-cum, 0.99, method="linear"))
    assert ten_day == pytest.approx(expected)


def test_horizon_one_matches_plain_one_day():
    # horizon=1 must reduce exactly to the ordinary one-day VaR.
    returns = np.random.default_rng(2).normal(0, 0.01, 1000)
    h1 = HistoricalVar(0.99, horizon=1).run(returns=returns).value
    plain = historical_var(returns, 0.99)
    assert h1 == pytest.approx(plain)


def test_horizon_var_is_not_sqrt_of_time_scaling():
    # The 10-day VaR should differ from the naive sqrt-of-time shortcut, because
    # it is built from the real h-day distribution (which is fatter/thinner than
    # a scaled one-day number unless the returns are exactly i.i.d. normal).
    returns = np.random.default_rng(7).normal(0, 0.01, 1000)
    one_day = HistoricalVar(0.99, horizon=1).run(returns=returns).value
    ten_day = HistoricalVar(0.99, horizon=10).run(returns=returns).value
    assert ten_day != pytest.approx(one_day * np.sqrt(10), rel=1e-3)


def test_es_is_mean_of_losses_beyond_var():
    # Losses 1..100; at 95% confidence the VaR is ~95, so ES is the mean of the
    # losses at or beyond it (96..100 -> 98).
    returns = -np.arange(1.0, 101.0)  # losses 1..100
    var = historical_var(returns, 0.95)
    es = historical_es(returns, 0.95, var)
    tail = np.arange(1.0, 101.0)
    expected = tail[tail >= var].mean()
    assert es == pytest.approx(expected)


def test_es_is_at_least_var():
    returns = np.random.default_rng(0).normal(0, 0.02, 5000)
    result = HistoricalVar(0.99).run(returns=returns)
    assert result.expected_shortfall >= result.value


def test_es_is_traced_in_result():
    returns = np.random.default_rng(1).normal(0, 0.01, 1000)
    result = HistoricalVar(0.99).run(returns=returns)
    assert "es" in result.steps
    assert "es_threshold" in result.steps
    # The reported ES is the (one-day) ES computed by the model.
    assert result.expected_shortfall == pytest.approx(result.steps["es"])


def test_es_grows_with_horizon_but_not_by_sqrt_of_time():
    # The h-day ES is the average loss beyond the h-day VaR, read off the real
    # h-day distribution. It should be larger than the one-day ES, but it need
    # not equal the sqrt-of-time scaling of it.
    returns = np.random.default_rng(2).normal(0, 0.01, 1000)
    one = HistoricalVar(0.99, horizon=1).run(returns=returns).expected_shortfall
    ten = HistoricalVar(0.99, horizon=10).run(returns=returns).expected_shortfall
    assert ten > one
    # In the right ballpark of sqrt(10) for near-i.i.d. data, but not exact.
    assert 2.0 < ten / one < 4.5

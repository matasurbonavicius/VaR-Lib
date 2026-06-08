"""Tests for the Engle-Manganelli Dynamic Quantile backtest."""

import numpy as np
import pytest

from varlib.backtest import dynamic_quantile_test


def test_well_specified_model_passes():
    # Independent breaches at exactly the expected rate must NOT be rejected.
    rng = np.random.default_rng(11)
    breaches = (rng.uniform(0, 1, 3000) < 0.01).astype(float)
    result = dynamic_quantile_test(breaches, 0.99, n_lags=4)
    assert not result.reject
    assert result.p_value > 0.05


def test_clustered_breaches_are_rejected():
    # Breaches that follow breaches (a Markov-clustered series) are predictable
    # from their own lags, so the DQ test must reject.
    rng = np.random.default_rng(12)
    breaches = np.zeros(3000)
    state = 0
    for i in range(3000):
        p = 0.5 if state else 0.005
        state = 1 if rng.uniform() < p else 0
        breaches[i] = state
    result = dynamic_quantile_test(breaches, 0.99, n_lags=4)
    assert result.reject
    assert result.p_value < 0.05


def test_p_value_in_unit_interval():
    rng = np.random.default_rng(13)
    breaches = (rng.uniform(0, 1, 1000) < 0.05).astype(float)
    result = dynamic_quantile_test(breaches, 0.95, n_lags=3)
    assert 0.0 <= result.p_value <= 1.0


def test_var_regressor_adds_a_degree_of_freedom():
    # Including the VaR forecast adds one regressor (one more dof).
    rng = np.random.default_rng(14)
    breaches = (rng.uniform(0, 1, 1000) < 0.01).astype(float)
    forecasts = np.full(1000, 0.03)
    without = dynamic_quantile_test(breaches, 0.99, n_lags=4)
    with_var = dynamic_quantile_test(breaches, 0.99, var_forecasts=forecasts, n_lags=4)
    assert with_var.n_regressors == without.n_regressors + 1


def test_audit_keys_present():
    rng = np.random.default_rng(15)
    breaches = (rng.uniform(0, 1, 800) < 0.01).astype(float)
    result = dynamic_quantile_test(breaches, 0.99, n_lags=4)
    for key in ["hits_mean", "coefficients", "statistic", "p_value", "reject"]:
        assert key in result.steps


def test_rejects_too_few_observations():
    with pytest.raises(ValueError):
        dynamic_quantile_test(np.array([0.0, 1.0, 0.0]), 0.99, n_lags=4)

"""Tests for Historical Bootstrap VaR."""

import numpy as np
import pytest

from varlib import HistoricalBootstrapVar, historical_bootstrap_var, historical_var


def test_bootstrap_is_close_to_plain_historical():
    rng = np.random.default_rng(3)
    returns = rng.normal(0, 0.02, 4000)
    plain = historical_var(returns, 0.99)
    boot = historical_bootstrap_var(returns, 0.99, n_resamples=2000, seed=0)
    # The bootstrap mean should sit close to the plain estimate.
    assert boot == pytest.approx(plain, rel=0.1)


def test_reports_standard_error():
    rng = np.random.default_rng(4)
    returns = rng.normal(0, 0.02, 1000)
    steps = {}
    historical_bootstrap_var(returns, 0.99, n_resamples=500, seed=0, steps=steps)
    assert steps["var_std_error"] > 0
    assert steps["resample_vars"].size == 500


def test_is_reproducible_with_seed():
    returns = np.random.default_rng(5).normal(0, 0.02, 800)
    a = historical_bootstrap_var(returns, 0.99, n_resamples=300, seed=42)
    b = historical_bootstrap_var(returns, 0.99, n_resamples=300, seed=42)
    assert a == b


def test_model_wrapper_runs():
    returns = np.random.default_rng(6).normal(0, 0.01, 1000)
    result = HistoricalBootstrapVar(0.99, n_resamples=200).run(returns=returns)
    assert result.method == "historical_bootstrap"
    assert result.value > 0


def test_rejects_bad_resample_count():
    with pytest.raises(ValueError):
        HistoricalBootstrapVar(0.99, n_resamples=0)


def test_bootstrap_es_is_close_to_plain_es():
    rng = np.random.default_rng(50)
    returns = rng.normal(0, 0.02, 4000)
    from varlib import historical_es
    plain = historical_es(returns, 0.99)
    result = HistoricalBootstrapVar(0.99, n_resamples=2000, seed=0).run(returns=returns)
    assert result.expected_shortfall == pytest.approx(plain, rel=0.1)


def test_bootstrap_reports_es_standard_error():
    rng = np.random.default_rng(51)
    returns = rng.normal(0, 0.02, 1000)
    result = HistoricalBootstrapVar(0.99, n_resamples=500, seed=0).run(returns=returns)
    assert result.steps["es_std_error"] > 0
    assert result.steps["resample_es"].size == 500


def test_bootstrap_es_at_least_var():
    returns = np.random.default_rng(52).normal(0, 0.02, 2000)
    result = HistoricalBootstrapVar(0.99, n_resamples=300).run(returns=returns)
    assert result.expected_shortfall >= result.value

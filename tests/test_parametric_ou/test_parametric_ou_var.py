"""Tests for the OU VaR calculation itself."""

import numpy as np
import pytest

from varlib import ParametricOuVar, parametric_ou_var
from varlib.models.parametric.ou import OuParameters
from tests.test_parametric_ou.test_estimate_ou_parameters import simulate_ou


def test_var_is_positive_and_traced():
    series = simulate_ou(kappa=0.2, theta=0.0, sigma=0.02, n=5000, seed=10)
    steps = {}
    var = parametric_ou_var(series, 0.99, steps=steps)
    assert var > 0
    assert {"kappa", "theta", "simulated_returns", "var"} <= set(steps)


def test_conditional_var_depends_on_last_value():
    # Starting far above the mean, the cumulative-return expectation reverts down,
    # so the downside (and thus VaR) differs from starting far below the mean.
    base = simulate_ou(kappa=0.5, theta=0.0, sigma=0.02, n=5000, seed=11)
    params = OuParameters(kappa=0.5, theta=0.0, sigma=0.02, b=np.exp(-0.5), last_value=0.1)
    high = parametric_ou_var(base, 0.99, params=params)
    params_low = params._replace(last_value=-0.1)
    low = parametric_ou_var(base, 0.99, params=params_low)
    assert high != low


def test_override_parameters_are_used():
    series = simulate_ou(kappa=0.3, theta=0.0, sigma=0.01, n=1000, seed=12)
    params = OuParameters(kappa=0.3, theta=0.0, sigma=0.05, b=np.exp(-0.3), last_value=0.0)
    steps = {}
    parametric_ou_var(series, 0.99, params=params, steps=steps)
    assert steps["params_source"] == "override"
    assert steps["sigma_ou"] == 0.05


def test_model_wrapper_runs():
    series = simulate_ou(kappa=0.2, theta=0.0, sigma=0.02, n=3000, seed=13)
    result = ParametricOuVar(0.99).run(returns=series)
    assert result.method == "parametric_ou"
    assert result.value > 0


def test_es_greater_than_var_and_traced():
    series = simulate_ou(kappa=0.3, theta=0.0, sigma=0.02, n=5000, seed=14)
    result = ParametricOuVar(0.99).run(returns=series)
    assert result.expected_shortfall > result.value
    assert "es" in result.steps
    assert "simulated_returns" in result.steps


def test_es_matches_cumulative_gaussian_form():
    # With known params at horizon 1, S_1 = x_1 has mean 0 (last_value == theta)
    # and std sigma, so the Gaussian ES is sigma * pdf(z) / 0.01. The model is now
    # Monte Carlo, so it recovers this within sampling error rather than exactly.
    from varlib.models.parametric.brownian import normal_quantile, normal_pdf
    params = OuParameters(kappa=0.5, theta=0.0, sigma=0.02,
                          b=np.exp(-0.5), last_value=0.0)
    z = normal_quantile(0.01)
    expected = 0.02 * normal_pdf(z) / 0.01
    result = ParametricOuVar(0.99, params=params).run(returns=np.zeros(10))
    assert result.expected_shortfall == pytest.approx(expected, rel=0.05)

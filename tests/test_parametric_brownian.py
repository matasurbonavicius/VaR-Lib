"""Tests for Parametric Brownian (Gaussian) VaR, including the normal quantile."""

import numpy as np
import pytest

from varlib import ParametricBrownianVar, parametric_brownian_var
from varlib.models.parametric.brownian import normal_quantile


def test_normal_quantile_known_values():
    # Standard textbook z-scores.
    assert normal_quantile(0.5) == pytest.approx(0.0, abs=1e-6)
    assert normal_quantile(0.975) == pytest.approx(1.95996, abs=1e-4)
    assert normal_quantile(0.025) == pytest.approx(-1.95996, abs=1e-4)
    assert normal_quantile(0.01) == pytest.approx(-2.32635, abs=1e-4)


def test_normal_quantile_is_symmetric():
    for p in [0.001, 0.05, 0.2, 0.4]:
        assert normal_quantile(p) == pytest.approx(-normal_quantile(1 - p), abs=1e-6)


def test_normal_quantile_rejects_out_of_range():
    for bad in [0.0, 1.0, -0.1, 1.5]:
        with pytest.raises(ValueError):
            normal_quantile(bad)


def test_var_matches_closed_form_for_known_mu_sigma():
    # With mu=0, sigma=0.02, the 99% VaR is -(0 + z*sigma) = 2.32635 * 0.02.
    var = parametric_brownian_var(np.zeros(10), 0.99, mu=0.0, sigma=0.02)
    assert var == pytest.approx(2.32635 * 0.02, abs=1e-4)


def test_estimates_mu_and_sigma_when_not_given():
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, 50_000)
    steps = {}
    parametric_brownian_var(returns, 0.99, steps=steps)
    assert steps["mu"] == pytest.approx(0.001, abs=2e-4)
    assert steps["sigma"] == pytest.approx(0.02, abs=2e-4)
    assert steps["mu_source"] == "estimated"


def test_gaussian_var_close_to_historical_on_normal_data():
    rng = np.random.default_rng(8)
    returns = rng.normal(0, 0.02, 100_000)
    from varlib import historical_var
    g = parametric_brownian_var(returns, 0.99)
    h = historical_var(returns, 0.99)
    assert g == pytest.approx(h, rel=0.05)


def test_model_wrapper_runs():
    returns = np.random.default_rng(9).normal(0, 0.01, 1000)
    result = ParametricBrownianVar(0.99).run(returns=returns)
    assert result.method == "parametric_brownian"
    assert result.value > 0


def test_es_matches_gaussian_closed_form():
    # For N(0, sigma) at 99%, ES = sigma * pdf(z) / 0.01, z = quantile(0.01).
    from varlib.models.parametric.brownian import normal_quantile, normal_pdf
    sigma = 0.02
    z = normal_quantile(0.01)
    expected = sigma * normal_pdf(z) / 0.01
    result = ParametricBrownianVar(0.99, mu=0.0, sigma=sigma).run(returns=np.zeros(10))
    assert result.expected_shortfall == pytest.approx(expected, abs=1e-6)


def test_es_greater_than_var():
    # ES must strictly exceed VaR for a continuous distribution.
    result = ParametricBrownianVar(0.99, mu=0.0, sigma=0.02).run(returns=np.zeros(10))
    assert result.expected_shortfall > result.value


def test_es_close_to_historical_es_on_normal_data():
    rng = np.random.default_rng(40)
    returns = rng.normal(0, 0.02, 100_000)
    from varlib import historical_es
    g = ParametricBrownianVar(0.99).run(returns=returns).expected_shortfall
    h = historical_es(returns, 0.99)
    assert g == pytest.approx(h, rel=0.05)


def test_normal_pdf_known_values():
    from varlib.models.parametric.brownian import normal_pdf
    assert normal_pdf(0.0) == pytest.approx(0.3989422804, abs=1e-9)
    assert normal_pdf(1.0) == pytest.approx(0.2419707245, abs=1e-9)

"""Tests for the Merton jump-diffusion VaR calculation."""

import numpy as np
import pytest

from varlib import ParametricJumpVar, parametric_jump_var
from varlib.models.parametric_jump import JumpParameters


def test_var_is_positive_and_reproducible():
    rng = np.random.default_rng(30)
    returns = rng.normal(0, 0.01, 5000)
    a = parametric_jump_var(returns, 0.99, n_simulations=20_000, seed=0)
    b = parametric_jump_var(returns, 0.99, n_simulations=20_000, seed=0)
    assert a > 0
    assert a == b  # same seed -> identical result


def test_jumps_increase_tail_risk_vs_gaussian():
    # A series with fat-tailed jumps should produce a larger VaR under the jump
    # model than a pure Gaussian fit, which ignores the jumps.
    rng = np.random.default_rng(31)
    returns = rng.normal(0, 0.01, 8000)
    idx = rng.choice(8000, size=80, replace=False)
    returns[idx] -= 0.08
    from varlib import parametric_brownian_var
    jump = parametric_jump_var(returns, 0.99, n_simulations=80_000, seed=1)
    gauss = parametric_brownian_var(returns, 0.99)
    assert jump > gauss


def test_override_parameters_skip_estimation():
    params = JumpParameters(
        mu_diffusion=0.0, sigma_diffusion=0.01,
        lambda_jump=0.05, mu_jump=-0.05, sigma_jump=0.02,
    )
    steps = {}
    parametric_jump_var(np.zeros(10), 0.99, params=params,
                        n_simulations=20_000, seed=2, steps=steps)
    assert steps["params_source"] == "override"
    assert steps["lambda_jump"] == 0.05


def test_traces_simulation_intermediates():
    rng = np.random.default_rng(32)
    returns = rng.normal(0, 0.01, 2000)
    steps = {}
    parametric_jump_var(returns, 0.99, n_simulations=10_000, seed=0, steps=steps)
    assert "simulated_returns" in steps
    assert steps["simulated_returns"].size == 10_000
    assert "var" in steps


def test_model_wrapper_runs():
    returns = np.random.default_rng(33).normal(0, 0.01, 2000)
    result = ParametricJumpVar(0.99, n_simulations=10_000).run(returns=returns)
    assert result.method == "parametric_jump"
    assert result.value > 0


def test_es_greater_than_var_and_traced():
    returns = np.random.default_rng(34).normal(0, 0.01, 3000)
    result = ParametricJumpVar(0.99, n_simulations=50_000, seed=0).run(returns=returns)
    assert result.expected_shortfall >= result.value
    assert "es" in result.steps


def test_es_reproducible_with_seed():
    returns = np.random.default_rng(35).normal(0, 0.01, 3000)
    a = ParametricJumpVar(0.99, n_simulations=20_000, seed=7).run(returns=returns)
    b = ParametricJumpVar(0.99, n_simulations=20_000, seed=7).run(returns=returns)
    assert a.expected_shortfall == b.expected_shortfall


def test_jumps_raise_es_above_gaussian():
    rng = np.random.default_rng(36)
    returns = rng.normal(0, 0.01, 8000)
    idx = rng.choice(8000, size=80, replace=False)
    returns[idx] -= 0.08
    from varlib import parametric_brownian_var_es
    jump_es = ParametricJumpVar(0.99, n_simulations=80_000, seed=1).run(
        returns=returns).expected_shortfall
    _, gauss_es = parametric_brownian_var_es(returns, 0.99)
    assert jump_es > gauss_es

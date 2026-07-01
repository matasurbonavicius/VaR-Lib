"""Tests for the jump/diffusion separation and parameter estimation."""

import numpy as np
import pytest

from varlib.models.parametric.jump import estimate_jump_parameters


def test_no_jumps_when_data_is_clean_normal():
    rng = np.random.default_rng(20)
    returns = rng.normal(0, 0.01, 10_000)
    params = estimate_jump_parameters(returns, jump_threshold=4.0)
    # With a 4-sigma threshold and clean normal data, very few/no jumps.
    assert params.lambda_jump < 0.01


def test_detects_injected_jumps():
    rng = np.random.default_rng(21)
    returns = rng.normal(0, 0.005, 5000)
    # Inject 50 clear downward jumps.
    idx = rng.choice(5000, size=50, replace=False)
    returns[idx] -= 0.10
    steps = {}
    params = estimate_jump_parameters(returns, jump_threshold=3.0, steps=steps)
    assert steps["n_jumps"] >= 40           # most injected jumps detected
    assert params.lambda_jump == pytest.approx(0.01, abs=0.005)
    assert params.mu_jump < 0               # jumps were downward


def test_diffusion_excludes_the_jumps():
    rng = np.random.default_rng(22)
    returns = rng.normal(0, 0.01, 5000)
    returns[100] = 0.5  # one giant jump
    params = estimate_jump_parameters(returns, jump_threshold=3.0)
    # Diffusion sigma should stay near 0.01, not be inflated by the 0.5 outlier.
    assert params.sigma_diffusion == pytest.approx(0.01, abs=0.002)


def test_traces_intermediates():
    rng = np.random.default_rng(23)
    returns = rng.normal(0, 0.01, 1000)
    steps = {}
    estimate_jump_parameters(returns, steps=steps)
    assert {"robust_centre", "robust_sigma", "n_jumps",
            "mu_diffusion", "sigma_diffusion", "lambda_jump"} <= set(steps)


def test_requires_minimum_length():
    with pytest.raises(ValueError):
        estimate_jump_parameters(np.array([0.1, 0.2, 0.3]))

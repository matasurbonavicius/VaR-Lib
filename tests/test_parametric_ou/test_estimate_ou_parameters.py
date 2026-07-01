"""Tests for the OU parameter estimator (the AR(1) calibration step)."""

import numpy as np
import pytest

from varlib.models.parametric.ou import estimate_ou_parameters


def simulate_ou(kappa, theta, sigma, n, seed=0, x0=None):
    """Simulate an OU series with known parameters for recovery tests."""
    rng = np.random.default_rng(seed)
    b = np.exp(-kappa)
    x = np.empty(n)
    x[0] = theta if x0 is None else x0
    for t in range(1, n):
        x[t] = theta + b * (x[t - 1] - theta) + sigma * rng.normal()
    return x


def test_recovers_known_parameters():
    series = simulate_ou(kappa=0.1, theta=0.05, sigma=0.01, n=50_000, seed=1)
    params = estimate_ou_parameters(series)
    assert params.kappa == pytest.approx(0.1, abs=0.02)
    assert params.theta == pytest.approx(0.05, abs=0.005)
    assert params.sigma == pytest.approx(0.01, abs=0.001)


def test_slope_between_zero_and_one_for_mean_reverting():
    series = simulate_ou(kappa=0.3, theta=0.0, sigma=0.02, n=10_000, seed=2)
    params = estimate_ou_parameters(series)
    assert 0.0 < params.b < 1.0


def test_traces_regression_intermediates():
    series = simulate_ou(kappa=0.2, theta=0.0, sigma=0.01, n=2000, seed=3)
    steps = {}
    estimate_ou_parameters(series, steps)
    assert {"ar1_slope_b", "ar1_intercept_a", "residual_sigma", "kappa", "theta"} <= set(steps)


def test_near_random_walk_gives_tiny_kappa():
    # A random walk has slope just below 1, so mean reversion is ~zero. The
    # estimator must stay finite and report a near-zero reversion speed.
    rng = np.random.default_rng(4)
    walk = np.cumsum(rng.normal(0, 1, 5000))
    params = estimate_ou_parameters(walk)
    assert params.kappa < 0.01
    assert np.isfinite(params.theta)


def test_falls_back_when_slope_is_explosive():
    # If the fitted slope is >= 1 (no stationarity), kappa = -ln(b) would be
    # non-positive, so the estimator falls back to kappa = 0 and the sample mean.
    # Build a series whose AR(1) slope exceeds 1 (mildly explosive).
    n = 2000
    rng = np.random.default_rng(5)
    x = np.empty(n)
    x[0] = 1.0
    for t in range(1, n):
        x[t] = 1.001 * x[t - 1] + 1e-6 * rng.normal()
    params = estimate_ou_parameters(x)
    assert params.b >= 1.0
    assert params.kappa == 0.0  # fallback path
    assert np.isfinite(params.theta)


def test_requires_minimum_length():
    with pytest.raises(ValueError):
        estimate_ou_parameters(np.array([0.1, 0.2]))

"""Tests for EWMA / RiskMetrics VaR."""

import numpy as np
import pytest

from varlib import EwmaVar, ewma_var, ewma_var_es


def test_half_life_matches_lambda():
    # With lambda=0.94 the weight half-life is ln(0.5)/ln(0.94) ~ 11.2 days.
    steps = {}
    ewma_var(np.random.default_rng(1).normal(0, 0.01, 1000), 0.99, 0.94, steps=steps)
    assert steps["half_life_days"] == pytest.approx(11.2, abs=0.2)
    assert steps["weight_on_last_day"] == pytest.approx(0.06, abs=1e-9)


def test_weights_decay_geometrically_to_recover_known_sigma():
    # If returns are constant magnitude, the EWMA sigma must converge to that
    # magnitude regardless of lambda (the weights sum to 1).
    r = np.full(2000, 0.0)
    r[1::2] = 0.02
    r[0::2] = -0.02            # |r| == 0.02 everywhere, mean ~ 0
    steps = {}
    ewma_var(r, 0.99, 0.94, mu=0.0, steps=steps)
    assert steps["sigma"] == pytest.approx(0.02, abs=1e-3)


def test_volatility_reacts_to_recent_regime():
    # A calm history followed by a turbulent block should lift the EWMA sigma far
    # above the calm-only estimate -- the whole point of recency weighting.
    rng = np.random.default_rng(2)
    calm = rng.normal(0, 0.005, 500)
    storm = np.concatenate([calm, rng.normal(0, 0.05, 30)])
    s_calm, s_storm = {}, {}
    ewma_var(calm, 0.99, 0.94, steps=s_calm)
    ewma_var(storm, 0.99, 0.94, steps=s_storm)
    assert s_storm["sigma"] > 2 * s_calm["sigma"]


def test_matches_gaussian_when_lambda_makes_it_flat():
    # EWMA with a very persistent lambda and zero drift gives a Gaussian-style VaR
    # at the EWMA volatility; VaR and ES must satisfy ES > VaR.
    rng = np.random.default_rng(3)
    result = EwmaVar(0.99, lambda_=0.97).run(returns=rng.normal(0, 0.02, 3000))
    assert result.expected_shortfall > result.value
    assert result.value > 0


def test_rejects_bad_lambda():
    for bad in [0.0, 1.0, -0.1, 1.5]:
        with pytest.raises(ValueError):
            EwmaVar(0.99, lambda_=bad)


def test_traced_steps_present():
    steps = {}
    ewma_var_es(np.random.default_rng(4).normal(0, 0.02, 1000), 0.99, steps=steps)
    for key in ["lambda", "ewma_variance", "sigma", "z_score", "var", "es"]:
        assert key in steps


def test_model_wrapper_runs():
    result = EwmaVar(0.99).run(returns=np.random.default_rng(5).normal(0, 0.01, 1000))
    assert result.method == "ewma"
    assert result.value > 0

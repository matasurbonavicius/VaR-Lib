"""Tests for multi-day (h-day) horizon VaR across all models.

The horizon is built into each model: the VaR is computed directly at the
holding period (the quantile of the h-day loss distribution, the OU h-step
conditional law, a simulated h-day path, ...), NOT a one-day number scaled by
sqrt(h). These tests pin that behaviour down.
"""

import numpy as np
import pytest

from varlib import (
    HistoricalVar,
    HistoricalBootstrapVar,
    ParametricBrownianVar,
    ParametricOuVar,
    ParametricJumpVar,
)
from varlib.base import overlapping_cumulative_returns


# -- the cumulative-returns helper -----------------------------------------


def test_overlapping_cumulative_returns_sums_windows():
    r = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    out = overlapping_cumulative_returns(r, 3)
    # windows: 0.1+0.2+0.3, 0.2+0.3+0.4, 0.3+0.4+0.5
    assert np.allclose(out, [0.6, 0.9, 1.2])
    assert out.size == r.size - 3 + 1


def test_overlapping_cumulative_returns_horizon_one_is_identity():
    r = np.array([0.1, -0.2, 0.3])
    assert np.allclose(overlapping_cumulative_returns(r, 1), r)


def test_overlapping_cumulative_returns_needs_enough_data():
    with pytest.raises(ValueError):
        overlapping_cumulative_returns(np.array([0.1, 0.2]), 5)


# -- horizon=1 must reduce to the one-day result for every model ------------


def _models_h(h):
    return [
        HistoricalVar(0.99, horizon=h),
        HistoricalBootstrapVar(0.99, horizon=h, n_resamples=200, seed=0),
        ParametricBrownianVar(0.99, horizon=h),
        ParametricOuVar(0.99, horizon=h),
        ParametricJumpVar(0.99, horizon=h, n_simulations=20_000, seed=0),
    ]


def test_horizon_one_equals_default_for_all_models():
    returns = np.random.default_rng(0).normal(0.0003, 0.012, 1000)
    for default, h1 in zip(_models_h(1), _models_h(1)):
        # Two independently-built horizon-1 models agree (sanity / determinism).
        assert default.run(returns=returns).value == pytest.approx(
            h1.run(returns=returns).value
        )


def test_ten_day_var_larger_than_one_day_for_diffusive_models():
    # For (near) i.i.d. data the 10-day VaR is materially larger than the 1-day.
    returns = np.random.default_rng(1).normal(0.0003, 0.012, 1500)
    for mk in (
        lambda h: HistoricalVar(0.99, horizon=h),
        lambda h: ParametricBrownianVar(0.99, horizon=h),
        lambda h: ParametricJumpVar(0.99, horizon=h, n_simulations=20_000, seed=0),
    ):
        v1 = mk(1).run(returns=returns).value
        v10 = mk(10).run(returns=returns).value
        assert v10 > 1.5 * v1


# -- Historical: direct quantile of cumulative returns ----------------------


def test_historical_ten_day_matches_manual_cumulative_quantile():
    returns = np.random.default_rng(2).normal(0, 0.01, 800)
    v10 = HistoricalVar(0.99, horizon=10).run(returns=returns).value
    cum = overlapping_cumulative_returns(returns, 10)
    assert v10 == pytest.approx(float(np.quantile(-cum, 0.99, method="linear")))


# -- Bootstrap: each resample sums `horizon` daily draws --------------------


def test_bootstrap_ten_day_close_to_direct_historical_ten_day():
    # The bootstrapped 10-day VaR should sit near the plain historical 10-day VaR
    # (both describe the same 10-day loss distribution).
    returns = np.random.default_rng(3).normal(0, 0.015, 3000)
    boot10 = HistoricalBootstrapVar(
        0.99, horizon=10, n_resamples=1500, seed=0
    ).run(returns=returns).value
    hist10 = HistoricalVar(0.99, horizon=10).run(returns=returns).value
    assert boot10 == pytest.approx(hist10, rel=0.15)


# -- OU: cumulative h-day return; variance grows but slower than i.i.d. -----


def _ou_cumulative_var_std(b, sig, theta, x0, h, conf=0.99):
    """Reference closed form for the std of the cumulative AR(1) sum S_h."""
    j = np.arange(1, h + 1)
    loadings = (1.0 - b ** (h - j + 1)) / (1.0 - b)
    return np.sqrt(sig * sig * np.sum(loadings * loadings))


def _fit_ar1(x):
    xp, xc = x[:-1], x[1:]
    b = np.cov(xp, xc, ddof=0)[0, 1] / np.var(xp)
    a = xc.mean() - b * xp.mean()
    resid = xc - (a + b * xp)
    sig = resid.std(ddof=2)
    theta = a / (1 - b) if 0 < b < 1 else xc.mean()
    return b, sig, theta, x[-1]


def test_ou_horizon_one_is_loss_on_next_single_return():
    # At horizon 1, S_1 = x_1: sigma_sum == residual sigma, no accumulation.
    rng = np.random.default_rng(2)
    n, b, sig = 2000, 0.6, 0.02
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = b * x[t - 1] + rng.normal(0, sig)
    res = ParametricOuVar(0.99, horizon=1).run(returns=x)
    assert res.steps["sigma_sum"] == pytest.approx(res.steps["sigma_ou"])


def test_ou_cumulative_std_matches_closed_form():
    # The model's sigma_sum equals the cumulative-AR(1) closed form.
    rng = np.random.default_rng(1)
    n, b_true, sig_true = 4000, 0.7, 0.01
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = b_true * x[t - 1] + rng.normal(0, sig_true)

    b, sig, theta, x0 = _fit_ar1(x)
    for h in (2, 5, 10, 50):
        res = ParametricOuVar(0.99, horizon=h).run(returns=x)
        expected = _ou_cumulative_var_std(b, sig, theta, x0, h)
        assert res.steps["sigma_sum"] == pytest.approx(expected, rel=1e-6)


def test_ou_cumulative_variance_direction_follows_autocorrelation():
    # The cumulative-sum variance vs the i.i.d. h*sigma^2 depends on the sign of
    # the AR(1) coefficient b:
    #   b > 0 (persistent): shocks compound -> Var[S_h] ABOVE  h*sigma^2.
    #   b < 0 (oscillating): shocks cancel  -> Var[S_h] BELOW  h*sigma^2.
    # This is exactly the structure sqrt-of-time scaling cannot capture.
    h = 10

    def sigma_sum_vs_iid(b_true):
        rng = np.random.default_rng(abs(int(b_true * 100)) + 1)
        n, sig = 4000, 0.01
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = b_true * x[t - 1] + rng.normal(0, sig)
        res = ParametricOuVar(0.99, horizon=h).run(returns=x)
        return res.steps["sigma_sum"], np.sqrt(h) * res.steps["sigma_ou"]

    pos_sum, pos_iid = sigma_sum_vs_iid(0.7)
    assert pos_sum > pos_iid                       # persistence amplifies

    neg_sum, neg_iid = sigma_sum_vs_iid(-0.7)
    assert neg_sum < neg_iid                        # oscillation damps

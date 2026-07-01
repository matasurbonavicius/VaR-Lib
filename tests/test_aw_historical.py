"""Tests for Age-weighted Historical VaR (BRW)."""

import numpy as np
import pytest

from varlib import AgeWeightedHistoricalVar, HistoricalVar
from varlib.base import VarResult
from varlib.models.non_parametric.aw_historical_simulation import _weighted_var_es


def test_reproduces_brw_paper_example():
    # BRW (1998) "The Best of Both Worlds", pp. 6-7 worked example: lambda=0.98,
    # K=100, 5% VaR. The paper's half-weight rule puts -2.70% at the 4.78th
    # percentile and the interpolated 5% VaR at ~2.63%. We feed the six tail
    # observations from the paper's table (plus filler mass at zero loss to make
    # the 100-day weights sum to 1) and check we land on the same tail number.
    lam, K = 0.98, 100
    returns = np.array([-3.30, -2.90, -2.70, -2.50, -2.40, -2.30]) / 100.0
    periods_ago = np.array([3, 2, 65, 45, 5, 30])
    const = (1 - lam) / (1 - lam**K)
    weights = const * lam ** (periods_ago - 1)

    # -2.70% should sit at the paper's stated 4.78th percentile.
    losses = -returns
    order = np.argsort(losses)[::-1]
    centered = (np.cumsum(weights[order]) - weights[order]) + 0.5 * weights[order]
    assert centered[2] == pytest.approx(0.0478, abs=5e-4)

    filler = 1.0 - weights.sum()
    r = np.concatenate([returns, [0.0]])
    w = np.concatenate([weights, [filler]])
    var, _ = _weighted_var_es(r, w, confidence=0.95)
    # ~2.63% in the paper; 2.65% interpolating between the two real adjacent
    # table rows (the paper's prose uses a phantom -2.60% row not in its table).
    assert var == pytest.approx(0.0263, abs=3e-4)


def test_lambda_one_reduces_to_plain_historical():
    # With no decay the weights are flat, so this must match plain Historical VaR
    # exactly (up to the quantile convention: step-function vs interpolated).
    returns = np.random.default_rng(0).normal(0, 0.01, 2000)
    aw = AgeWeightedHistoricalVar(0.99, lambda_decay=1.0).run(returns=returns).value
    plain = HistoricalVar(0.99).run(returns=returns).value
    assert aw == pytest.approx(plain, rel=0.02)


def test_recent_crash_lifts_var_more_than_old_crash():
    # Same crash, placed at the newest end vs the oldest end of the window.
    calm = np.random.default_rng(1).normal(0, 0.005, 500)
    crash = np.full(20, -0.10)

    recent_crash = np.concatenate([calm, crash])   # crash at the newest end
    old_crash = np.concatenate([crash, calm])       # crash at the oldest end

    model = AgeWeightedHistoricalVar(0.95, lambda_decay=0.97)
    var_recent = model.run(returns=recent_crash).value
    var_old = model.run(returns=old_crash).value
    # Weighting recent days more, the recent crash should push VaR higher.
    assert var_recent > var_old


def test_higher_confidence_gives_higher_var():
    returns = np.random.default_rng(2).normal(0, 0.02, 3000)
    v95 = AgeWeightedHistoricalVar(0.95).run(returns=returns).value
    v99 = AgeWeightedHistoricalVar(0.99).run(returns=returns).value
    assert v99 > v95


def test_weights_are_normalised_and_favour_recent():
    returns = np.random.default_rng(3).normal(0, 0.01, 200)
    steps = AgeWeightedHistoricalVar(0.99, lambda_decay=0.95).run(returns=returns).steps
    weights = steps["weights"]
    assert weights.sum() == pytest.approx(1.0)
    # Weights are oldest-first, so the newest (last) weight is the largest.
    assert weights[-1] > weights[0]


def test_model_returns_var_result_with_full_trace():
    returns = np.random.default_rng(5).normal(0, 0.01, 1000)
    result = AgeWeightedHistoricalVar(confidence=0.99).run(returns=returns)
    assert isinstance(result, VarResult)
    assert result.method == "age_weighted_historical"
    assert result.value > 0
    assert set(result.steps) >= {"weights", "lambda_decay", "var", "es"}


def test_es_is_at_least_var():
    returns = np.random.default_rng(6).normal(0, 0.02, 3000)
    result = AgeWeightedHistoricalVar(0.99).run(returns=returns)
    assert result.expected_shortfall >= result.value


def test_horizon_var_uses_cumulative_returns():
    returns = np.random.default_rng(7).normal(0, 0.01, 1000)
    one = AgeWeightedHistoricalVar(0.99, horizon=1).run(returns=returns).value
    ten = AgeWeightedHistoricalVar(0.99, horizon=10).run(returns=returns).value
    # A 10-day loss is bigger than a one-day loss.
    assert ten > one


def test_rejects_bad_lambda():
    with pytest.raises(ValueError):
        AgeWeightedHistoricalVar(0.99, lambda_decay=0.0)
    with pytest.raises(ValueError):
        AgeWeightedHistoricalVar(0.99, lambda_decay=1.5)

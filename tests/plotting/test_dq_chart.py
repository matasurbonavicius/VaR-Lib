"""Tests for the Dynamic Quantile chart."""

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from varlib.backtest import dynamic_quantile_test
from varlib.plotting import dq_chart


def _dq_result(seed=0, n=400):
    rng = np.random.default_rng(seed)
    breaches = (rng.random(n) < 0.01).astype(float)
    var_forecasts = rng.normal(0.03, 0.005, n)
    return dynamic_quantile_test(breaches, 0.99, var_forecasts=var_forecasts)


def test_returns_axes():
    import matplotlib.axes

    ax = dq_chart(_dq_result())
    assert isinstance(ax, matplotlib.axes.Axes)


def test_one_bar_per_regressor():
    """A bar is drawn for the constant, each lag, and the VaR level."""
    result = _dq_result()
    ax = dq_chart(result)
    # barh produces one patch per regressor.
    assert len(ax.patches) == result.n_regressors
    # The y tick labels name every regressor.
    assert len(ax.get_yticklabels()) == result.n_regressors


def test_labels_include_var_when_used():
    ax = dq_chart(_dq_result())
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "constant" in labels
    assert "VaR level" in labels
    assert any(l.startswith("lag") for l in labels)


def test_works_without_var_regressor():
    """Omitting var_forecasts drops the VaR row but still plots."""
    rng = np.random.default_rng(1)
    breaches = (rng.random(300) < 0.01).astype(float)
    result = dynamic_quantile_test(breaches, 0.99)
    ax = dq_chart(result)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert "VaR level" not in labels
    assert len(ax.patches) == result.n_regressors

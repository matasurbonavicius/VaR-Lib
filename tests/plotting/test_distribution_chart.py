"""Tests for the return-distribution chart."""

import numpy as np

from varlib import HistoricalVar
from varlib.plotting import distribution_chart


def test_draws_var_and_es_lines():
    rng = np.random.default_rng(0)
    returns = rng.normal(0, 0.02, 2000)
    result = HistoricalVar(0.99).run(returns=returns)
    ax = distribution_chart(returns, result.value, result.expected_shortfall, 0.99)
    # Two vertical lines: VaR and ES.
    vlines = [ln for ln in ax.lines]
    assert len(vlines) >= 2
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert any("VaR" in lbl for lbl in labels)
    assert any("ES" in lbl for lbl in labels)


def test_es_line_is_left_of_var_line():
    # ES >= VaR, so on the return (loss) axis the ES line sits further left.
    rng = np.random.default_rng(1)
    returns = rng.normal(0, 0.02, 5000)
    result = HistoricalVar(0.99).run(returns=returns)
    ax = distribution_chart(returns, result.value, result.expected_shortfall, 0.99)
    var_x = ax.lines[0].get_xdata()[0]   # first line = VaR
    es_x = ax.lines[1].get_xdata()[0]    # second line = ES
    assert es_x <= var_x


def test_var_only_without_es():
    rng = np.random.default_rng(2)
    returns = rng.normal(0, 0.02, 1000)
    ax = distribution_chart(returns, 0.05, expected_shortfall=None)
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert not any("ES" in lbl for lbl in labels)

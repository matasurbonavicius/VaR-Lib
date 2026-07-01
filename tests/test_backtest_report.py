"""Tests for the high-level ``varlib.report`` workflow.

``run_backtest`` rolls a model and runs every test in one call, returning a
``BacktestReport``. These tests check that the bundled results match running the
pieces by hand, that the console view is faithful, and that the chart/save views
delegate correctly (matplotlib is a hard dependency, so they run unconditionally).
"""

import os

import numpy as np
import pandas as pd
import pytest

from varlib import HistoricalVar, ParametricJumpVar, ParametricOuVar, run_backtest
from varlib.report import BacktestReport, _prettify
from varlib.backtest import (
    basel_traffic_light,
    count_breaches,
    dynamic_quantile_test,
    kupiec_pof_test,
    rolling_backtest,
)


def _price_series(n=600, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=idx)


def test_run_backtest_returns_report_with_all_tests():
    report = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    assert isinstance(report, BacktestReport)
    assert report.confidence == 0.99
    assert report.summary.n_observations == len(report.losses) == len(report.forecasts)
    # Every verdict object is present.
    assert report.kupiec is not None
    assert report.dynamic_quantile is not None
    assert report.traffic_light is not None


def test_bundle_matches_running_the_pieces_by_hand():
    prices = _price_series()
    returns = np.diff(np.log(prices.to_numpy()))
    model = HistoricalVar(0.99)
    report = run_backtest(model, returns=returns, window=250)

    losses, forecasts, _ = rolling_backtest(model, returns=returns, window=250)
    summary = count_breaches(losses, forecasts)
    flags = np.asarray(summary.steps["is_breach"], dtype=float)

    assert report.summary.n_breaches == summary.n_breaches
    assert report.kupiec.p_value == kupiec_pof_test(flags, 0.99).p_value
    assert report.dynamic_quantile.p_value == (
        dynamic_quantile_test(flags, 0.99, var_forecasts=forecasts).p_value
    )
    assert report.traffic_light.zone == (
        basel_traffic_light(summary.n_breaches, summary.n_observations, 0.99).zone
    )


def test_confidence_comes_from_the_model():
    report = run_backtest(HistoricalVar(0.975), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    assert report.confidence == 0.975


def test_format_contains_every_verdict():
    report = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    text = report.format("My title")
    assert "My title" in text
    for label in ("Kupiec POF", "Dynamic Quantile", "Basel zone"):
        assert label in text
    assert "OK" in text or "REJECT" in text


def test_dashboard_returns_a_figure():
    import matplotlib
    matplotlib.use("Agg")

    report = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    fig = report.dashboard(title="t")
    import matplotlib.figure
    assert isinstance(fig, matplotlib.figure.Figure)


def test_save_writes_a_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    report = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    out = os.path.join(tmp_path, "dash.png")
    written = report.save(out, title="t")
    assert written and all(os.path.exists(p) for p in written)


# -- auto-generated default styling -----------------------------------------


def test_label_comes_from_the_model_class():
    assert run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250).label == "Historical"
    assert run_backtest(ParametricJumpVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250).label == "Parametric jump"
    assert run_backtest(ParametricOuVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250).label == "Parametric OU"


def test_prettify_falls_back_for_unknown_classes():
    # CamelCase split, trailing "Var" dropped, for a class not in the table.
    assert _prettify("MyFancyVar") == "My Fancy"
    assert _prettify("Whatever") == "Whatever"


def test_default_title_and_subtitle_describe_the_run():
    report = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=200)
    assert report.default_title() == "Historical VaR — backtest report"
    sub = report.default_subtitle()
    assert "1-day" in sub and "200-day window" in sub and "99% confidence" in sub


def test_default_footer_has_inputs_result_and_tests():
    report = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    footer = report.default_footer()
    categories = [row[0] for row in footer]
    assert categories == ["Inputs", "Result", "Tests"]
    inputs, result, tests = (row[1] for row in footer)
    assert "Historical" in inputs and "Window 250d" in inputs
    assert f"{report.summary.n_breaches}/{report.summary.n_observations}" in result
    assert "Kupiec POF" in tests and "Dynamic Quantile" in tests


def test_overlap_bias_note_only_for_multiday_overlapping():
    one_day = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    assert "biased" not in one_day.default_footer()[2][1]

    multi = run_backtest(
        HistoricalVar(0.99, horizon=10), returns=np.diff(np.log(_price_series().to_numpy())), window=250, overlap=True
    )
    assert "biased: overlapping windows" in multi.default_footer()[2][1]


def test_save_uses_generated_styling_when_caller_passes_nothing(tmp_path):
    import matplotlib
    matplotlib.use("Agg")

    report = run_backtest(HistoricalVar(0.99), returns=np.diff(np.log(_price_series().to_numpy())), window=250)
    written = report.save(os.path.join(tmp_path, "auto.pdf"))
    assert written and all(os.path.exists(p) for p in written)

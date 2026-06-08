"""Tests for the selective / multi-page report composer."""

import os

import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg")

from varlib.backtest import (  # noqa: E402
    basel_traffic_light,
    christoffersen_test,
    count_breaches,
    dynamic_quantile_test,
    kupiec_pof_test,
)
from varlib.plotting.report import (  # noqa: E402
    _paginate,
    _resolve_sections,
    build_report,
    save_report,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(4)
    losses = np.abs(rng.normal(0, 0.02, 400))
    forecasts = np.full(400, 0.05)
    return losses, forecasts


@pytest.fixture
def backtests(data):
    losses, forecasts = data
    summary = count_breaches(losses, forecasts)
    flags = np.array(summary.steps["is_breach"], dtype=float)
    return {
        "kupiec": kupiec_pof_test(flags, 0.99),
        "christoffersen": christoffersen_test(flags, 0.99),
        "dynamic_quantile": dynamic_quantile_test(flags, 0.99, var_forecasts=forecasts),
        "traffic_light": basel_traffic_light(summary.n_breaches, summary.n_observations, 0.99),
    }


def test_default_sections_fit_one_page():
    assert len(_paginate(_resolve_sections(None))) == 1


def test_all_sections_span_two_pages():
    assert len(_paginate(_resolve_sections("all"))) == 2


def test_unknown_section_rejected():
    with pytest.raises(ValueError):
        _resolve_sections(["breaches", "nonsense"])


def test_sections_normalised_to_reading_order():
    # Whatever order they are passed, the canonical order is preserved.
    out = _resolve_sections(["distribution", "breaches"])
    assert out == ["breaches", "distribution"]


def test_default_build_returns_single_figure(data):
    losses, forecasts = data
    fig = build_report(losses, forecasts, confidence=0.99)
    # A single Figure, not a list.
    assert hasattr(fig, "savefig")


def test_all_build_returns_two_figures(data, backtests):
    losses, forecasts = data
    figs = build_report(losses, forecasts, sections="all", confidence=0.99,
                        backtests=backtests)
    assert isinstance(figs, list)
    assert len(figs) == 2


def test_save_single_page_writes_one_file(data, tmp_path):
    losses, forecasts = data
    fig = build_report(losses, forecasts, confidence=0.99)
    path = os.path.join(tmp_path, "rep.pdf")
    written = save_report(fig, path)
    assert written == [path]
    assert os.path.exists(path)


def test_save_multipage_pdf_is_single_file(data, backtests, tmp_path):
    losses, forecasts = data
    figs = build_report(losses, forecasts, sections="all", confidence=0.99,
                        backtests=backtests)
    path = os.path.join(tmp_path, "rep.pdf")
    written = save_report(figs, path)
    assert written == [path]
    # The single PDF holds both pages.
    reader = pytest.importorskip("pypdf", reason="pypdf needed to count PDF pages")
    assert len(reader.PdfReader(path).pages) == 2


def test_save_multipage_png_writes_one_file_per_page(data, backtests, tmp_path):
    losses, forecasts = data
    figs = build_report(losses, forecasts, sections="all", confidence=0.99,
                        backtests=backtests)
    path = os.path.join(tmp_path, "rep.png")
    written = save_report(figs, path)
    assert len(written) == 2
    assert all(os.path.exists(p) for p in written)
    assert written[0].endswith("_p1.png")
    assert written[1].endswith("_p2.png")


def test_selective_sections_render(data, backtests, tmp_path):
    losses, forecasts = data
    fig = build_report(losses, forecasts, sections=["breaches", "tests"],
                       confidence=0.99, backtests=backtests)
    path = os.path.join(tmp_path, "sel.pdf")
    assert save_report(fig, path) == [path]

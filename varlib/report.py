"""
One call, the whole backtest: roll a model, run every test, render the report.

The backtest workflow has a fixed shape -- roll the model through history, count
the breaches, run the standard tests, then print or plot the verdict. That
shape was previously re-assembled by hand in every script. `BacktestReport`
packages it: hand it a model and a return series and it does the roll and every
test, then renders them to the console, a dashboard figure, or a PDF -- with a
title and a self-describing metrics footer generated for you, so the default
report is print-ready with no styling code at all.

    from varlib import HistoricalVar, run_backtest

    returns = np.log(prices / prices.shift(1)).dropna()
    report = run_backtest(HistoricalVar(0.99), returns=returns, window=250)
    report.print()                      # the console summary
    report.save("backtest.pdf")         # the print-ready dashboard, fully labelled

Every number the report shows is a plain field on the result objects it carries
(`report.kupiec`, `report.traffic_light`, ...), so nothing is hidden behind the
rendering. Pass your own ``title`` / ``subtitle`` / ``footer`` to override the
defaults.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from varlib.base import VarModel
from varlib.backtest import (
    BreachSummary,
    DynamicQuantileResult,
    KupiecResult,
    TrafficLightResult,
    basel_traffic_light,
    count_breaches,
    dynamic_quantile_test,
    kupiec_pof_test,
    rolling_backtest,
)

# Pretty display names for the known model classes. Anything not listed falls
# back to a generic prettifier, so new models still get a sensible label.
_MODEL_LABELS = {
    "HistoricalVar": "Historical",
    "HistoricalBootstrapVar": "Historical bootstrap",
    "ParametricBrownianVar": "Parametric Brownian",
    "ParametricOuVar": "Parametric OU",
    "ParametricJumpVar": "Parametric jump",
    "EwmaVar": "EWMA / RiskMetrics",
}


def _prettify(class_name: str) -> str:
    """Human label for a model class, e.g. ``ParametricOuVar`` -> ``Parametric OU``.

    Falls back to splitting CamelCase and dropping a trailing ``Var`` for any
    class not in :data:`_MODEL_LABELS`.
    """
    if class_name in _MODEL_LABELS:
        return _MODEL_LABELS[class_name]
    name = class_name[:-3] if class_name.endswith("Var") else class_name
    words = re.findall(r"[A-Z][a-z0-9]*", name) or [name]
    return " ".join(words)


@dataclass
class BacktestReport:
    """A rolled backtest plus its test verdicts, ready to print or plot.

    Holds the aligned ``(losses, forecasts, dates)`` series from the roll, the
    result object from each standard test, and the run parameters
    (model, window, horizon, overlap) needed to label a report. The renderers
    (``print`` / ``dashboard`` / ``report`` / ``save``) build a title and a
    self-describing footer from these on their own -- callers pass data, not
    styling.
    """

    confidence: float
    model_name: str
    window: int
    horizon: int
    overlap: bool
    losses: np.ndarray
    forecasts: np.ndarray
    dates: Sequence[Any]
    summary: BreachSummary
    kupiec: KupiecResult
    dynamic_quantile: DynamicQuantileResult
    traffic_light: TrafficLightResult

    @property
    def label(self) -> str:
        """The model's human-readable name, e.g. ``"Parametric jump"``."""
        return _prettify(self.model_name)

    # -- console ------------------------------------------------------------

    def print(self, title: Optional[str] = None) -> None:
        """Print the Kupiec / Dynamic Quantile / Basel summary to stdout."""
        print(self.format(title))

    def format(self, title: Optional[str] = None) -> str:
        """Return the console summary as a string (what ``print`` writes)."""
        c = self.confidence
        title = title or f"{self.label} VaR backtest"

        def verdict(reject: bool) -> str:
            return "REJECT" if reject else "OK"

        lines = [
            "=" * 64,
            f"{title}  (confidence = {c:.0%})",
            "=" * 64,
            f"Observations    : {self.summary.n_observations}",
            f"Breaches        : {self.summary.n_breaches} "
            f"(rate {self.summary.breach_rate * 100:.2f}%, "
            f"expected {(1 - c) * 100:.2f}%)",
            f"Kupiec POF      : p = {self.kupiec.p_value:.3f} "
            f"-> {verdict(self.kupiec.reject_at_5pct)}",
            f"Dynamic Quantile: p = {self.dynamic_quantile.p_value:.3f} "
            f"-> {verdict(self.dynamic_quantile.reject)}",
            f"Basel zone      : {self.traffic_light.zone.upper()} "
            f"(green<= {self.traffic_light.green_max}, "
            f"red>= {self.traffic_light.red_min})",
        ]
        return "\n".join(lines)

    # -- default report styling (built from the report's own contents) ------

    def default_title(self) -> str:
        """The report heading, e.g. ``"Parametric jump VaR — backtest report"``."""
        return f"{self.label} VaR — backtest report"

    def default_subtitle(self) -> str:
        """The grey line under the title: horizon, window, and confidence."""
        if self.horizon == 1:
            horizon_phrase = "1-day"
        else:
            scheme = "overlapping" if self.overlap else "non-overlapping"
            horizon_phrase = f"{self.horizon}-day ({scheme})"
        return (f"{horizon_phrase} horizon · rolling {self.window}-day window"
                f" · {self.confidence:.0%} confidence")

    def default_footer(self) -> list[tuple[str, str]]:
        """A labelled (category, value) grid summarising inputs, result, tests.

        Returns one row per category so the page lays them out as a clean grid
        and reads as a self-describing, reproducible record of the run -- no
        caller-supplied text required.
        """
        ok = lambda reject: "OK" if not reject else "REJECT"  # noqa: E731
        s, k = self.summary, self.kupiec
        dq, light = self.dynamic_quantile, self.traffic_light
        scheme = "overlapping" if self.overlap else "non-overlapping"
        span = self._date_span()

        # Overlapping multi-day windows are serially dependent, which biases the
        # independence-based tests; flag that on the page rather than hide it.
        bias_note = (" [biased: overlapping windows]"
                     if self.horizon > 1 and self.overlap else "")

        return [
            ("Inputs",
             f"Model {self.label}  ·  "
             f"Confidence {self.confidence:.1%}  ·  "
             f"Horizon {self.horizon}d ({scheme})  ·  "
             f"Window {self.window}d"
             + (f"  ·  Data {span}" if span else "")),
            ("Result",
             f"Breaches {s.n_breaches}/{s.n_observations}  ·  "
             f"Rate {s.breach_rate:.2%} (expected {1 - self.confidence:.2%})  ·  "
             f"Basel zone {light.zone.upper()}"),
            ("Tests",
             f"Kupiec POF {ok(k.reject_at_5pct)} (p={k.p_value:.3f})  ·  "
             f"Dynamic Quantile {ok(dq.reject)} (p={dq.p_value:.3f}){bias_note}"),
        ]

    def _date_span(self) -> str:
        """``'2020-01-02 – 2024-12-31 (1007 steps)'`` when dates are labelled."""
        n = len(self.dates)
        if not n:
            return ""
        first, last = self.dates[0], self.dates[-1]
        fmt = lambda d: d.date() if hasattr(d, "date") else d  # noqa: E731
        return f"{fmt(first)} – {fmt(last)} ({n} steps)"

    def _styling(self, kwargs: dict) -> dict:
        """Fill in title/subtitle/footer defaults for any the caller omitted."""
        kwargs.setdefault("title", self.default_title())
        kwargs.setdefault("subtitle", self.default_subtitle())
        kwargs.setdefault("footer", self.default_footer())
        return kwargs

    # -- charts (need matplotlib) -------------------------------------------

    def dashboard(self, **kwargs):
        """The four core charts on one A4 figure, titled and footed by default.

        Title, subtitle, and the metrics footer are generated from this report
        unless you pass your own. Other keyword arguments go to
        ``varlib.plotting.backtest_dashboard`` (e.g. ``a4=False`` for the
        on-screen layout).
        """
        from varlib.plotting import backtest_dashboard

        kwargs.setdefault("a4", True)
        return backtest_dashboard(
            self.losses, self.forecasts, dates=self.dates,
            confidence=self.confidence, **self._styling(kwargs),
        )

    def report(self, sections="all", **kwargs):
        """A selective, possibly multi-page report (Figure or list of Figures).

        Title/subtitle/footer default to this report's own; the ``"tests"`` panel
        uses its verdicts. Extra keyword arguments are forwarded to
        ``varlib.plotting.build_report``.
        """
        from varlib.plotting import build_report

        return build_report(
            self.losses, self.forecasts, sections=sections, dates=self.dates,
            confidence=self.confidence, backtests=self.backtests,
            **self._styling(kwargs),
        )

    @property
    def backtests(self) -> dict:
        """The test results keyed for ``build_report``'s ``backtests`` argument."""
        return {
            "kupiec": self.kupiec,
            "dynamic_quantile": self.dynamic_quantile,
            "traffic_light": self.traffic_light,
        }

    def save(self, path: str, *, sections=None, **kwargs) -> list[str]:
        """Render and write the report to ``path``; return the files written.

        With ``sections=None`` (default) this writes the one-page dashboard. Pass
        ``sections`` (e.g. ``"all"`` or ``["breaches", "tests"]``) for a
        selective report, which may span multiple pages. The format follows the
        file extension. Title, subtitle, and footer are generated by default;
        override any by passing it as a keyword argument.
        """
        from varlib.plotting import save_report

        if sections is None:
            return save_report(self.dashboard(**kwargs), path)
        return save_report(self.report(sections=sections, **kwargs), path)


def run_backtest(
    model: VarModel,
    *,
    returns: Any,
    window: int = 250,
    overlap: bool = True,
    dates: Optional[Sequence[Any]] = None,
) -> BacktestReport:
    """Roll ``model`` through history and run every backtest in one call.

    This is the headline workflow as a single function: it rolls the model
    (``varlib.backtest.rolling_backtest``), counts the breaches, and runs every
    standard test at the model's confidence level, returning a
    :class:`BacktestReport` that can print, plot, or save itself.

    Parameters
    ----------
    model
        Any ``VarModel`` instance. Its ``confidence`` is used for the tests and
        its ``horizon`` drives the realised-loss window.
    returns
        The log-return series, as in ``rolling_backtest``. A pandas Series labels
        each step from its index. Compute log returns from prices yourself first.
    window, overlap, dates
        Passed straight through to ``rolling_backtest``.
    """
    confidence = float(getattr(model, "confidence"))
    horizon = int(getattr(model, "horizon", 1))
    losses, forecasts, step_dates = rolling_backtest(
        model, returns=returns,
        window=window, overlap=overlap, dates=dates,
    )

    summary = count_breaches(losses, forecasts)
    flags = np.asarray(summary.steps["is_breach"], dtype=float)
    return BacktestReport(
        confidence=confidence,
        model_name=type(model).__name__,
        window=window,
        horizon=horizon,
        overlap=overlap,
        losses=losses,
        forecasts=forecasts,
        dates=step_dates,
        summary=summary,
        kupiec=kupiec_pof_test(flags, confidence),
        dynamic_quantile=dynamic_quantile_test(
            flags, confidence, var_forecasts=forecasts
        ),
        traffic_light=basel_traffic_light(
            summary.n_breaches, summary.n_observations, confidence
        ),
    )

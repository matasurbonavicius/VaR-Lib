"""
Backtest panel -- every test verdict as one clean table.

The individual charts show *how* a model behaves; this panel states the
*verdicts* in one block, the way a risk report leads: each test on its own row
with its statistic, its p-value, and a plain pass/fail. It pulls together the
backtests the library runs -- Kupiec POF, the Engle-Manganelli Dynamic Quantile
test, and the Basel traffic light -- so the reader sees the whole scorecard at a
glance.

It takes the already-computed result objects (so it never re-runs anything) and
draws them as a tidy, coloured table: green for a pass, red for a reject, with
the traffic-light row coloured by its actual zone.
"""

from __future__ import annotations

from typing import Optional

from varlib.backtest.dynamic_quantile import DynamicQuantileResult
from varlib.backtest.kupiec import KupiecResult
from varlib.backtest.traffic_light import TrafficLightResult
from varlib.plotting._style import COLORS, get_pyplot


def backtest_panel(
    kupiec: Optional[KupiecResult] = None,
    dynamic_quantile: Optional[DynamicQuantileResult] = None,
    traffic_light: Optional[TrafficLightResult] = None,
    title: str = "Backtest verdicts",
    ax=None,
):
    """
    Draw a compact table of backtest verdicts.

    Pass whichever result objects are available; each becomes one row. Rows are
    coloured by outcome -- green for a pass, red for a reject -- and the Basel row
    by its zone. Any test left as ``None`` is simply omitted.

    Parameters
    ----------
    kupiec, dynamic_quantile, traffic_light
        The result objects from the corresponding backtests.
    title
        Heading drawn above the table.
    ax
        Optional existing Axes. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = get_pyplot()

    # Build the rows: (test name, the metric, the verdict text, a colour key).
    # Colour key is "green" (pass), "red" (reject) or the literal Basel zone.
    rows: list[tuple[str, str, str, str]] = []

    if kupiec is not None:
        verdict = "REJECT" if kupiec.reject_at_5pct else "PASS"
        rows.append((
            "Kupiec POF",
            f"{kupiec.n_breaches}/{kupiec.n_observations}  ·  p = {kupiec.p_value:.3f}",
            verdict,
            "red" if kupiec.reject_at_5pct else "green",
        ))

    if dynamic_quantile is not None:
        verdict = "REJECT" if dynamic_quantile.reject else "PASS"
        rows.append((
            "Dynamic Quantile",
            f"DQ = {dynamic_quantile.statistic:.2f}  ·  "
            f"p = {dynamic_quantile.p_value:.3f}",
            verdict,
            "red" if dynamic_quantile.reject else "green",
        ))

    if traffic_light is not None:
        rows.append((
            "Basel traffic light",
            f"{traffic_light.n_breaches} breaches  ·  "
            f"green ≤ {traffic_light.green_max} · red ≥ {traffic_light.red_min}",
            traffic_light.zone.upper(),
            traffic_light.zone,
        ))

    if not rows:
        raise ValueError("Provide at least one backtest result to draw.")

    # The table is intentionally short: one tight line per test. Its natural
    # height is a small fixed band per row plus a little for the title -- it should
    # read as a compact scorecard, never balloon to fill whatever axes it is given.
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 0.42 + 0.30 * len(rows)))

    ax.set_axis_off()
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")

    # Lay the rows out top-to-bottom. Rather than spreading them across the whole
    # axes (which makes each row balloon on a tall subplot), we pin every row to a
    # FIXED pitch anchored at the top, so the table looks the same whether the axes
    # is short or tall -- any leftover height is simply left blank.
    row_h = 0.16                       # fixed row pitch, in axes fraction
    top = 0.90
    name_x, metric_x, verdict_x = 0.02, 0.42, 0.98

    for i, (name, metric, verdict, colour_key) in enumerate(rows):
        y = top - (i + 0.5) * row_h
        colour = COLORS.get(colour_key, COLORS["loss"])

        # A faint zebra band on alternate rows ties each row's columns together
        # without the heavy per-row colour wash the old layout used. It is sized to
        # the fixed pitch, so it stays a slim strip rather than a tall block.
        if i % 2 == 0:
            ax.axhspan(y - row_h * 0.5, y + row_h * 0.5, xmin=0.0, xmax=1.0,
                       color="#000000", alpha=0.035, zorder=0)

        ax.text(name_x, y, name, fontsize=9, fontweight="bold",
                va="center", ha="left", color="#222222")
        ax.text(metric_x, y, metric, fontsize=8, va="center", ha="left",
                color="#555555", family="monospace")
        # The verdict as a small, colour-coded chip, right-aligned -- a light
        # rounded box keeps it legible without the chip dominating the row.
        ax.text(verdict_x, y, verdict, fontsize=8, fontweight="bold",
                va="center", ha="right", color=colour,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=colour,
                          alpha=0.14, edgecolor="none"))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax

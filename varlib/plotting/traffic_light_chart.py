"""
Basel traffic-light chart -- the supervisory view of the breach count.

The Basel framework places a model in a zone based on how many breaches it had
over the window: green (fine), yellow (watch), red (rejected). This chart draws
those three zones as coloured bands along the breach-count axis and marks where
the actual breach count falls.
"""

from __future__ import annotations

from typing import Optional

from varlib.backtest.traffic_light import TrafficLightResult, basel_traffic_light
from varlib.plotting._style import COLORS, get_pyplot, style_axes


def traffic_light_chart(
    result: Optional[TrafficLightResult] = None,
    n_breaches: Optional[int] = None,
    n_observations: int = 250,
    confidence: float = 0.99,
    ax=None,
):
    """
    Draw the Basel traffic-light zones and mark the observed breach count.

    Pass either a `TrafficLightResult` (from `basel_traffic_light`) or the raw
    `n_breaches` / `n_observations` to compute one here.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = get_pyplot()

    if result is None:
        if n_breaches is None:
            raise ValueError("Provide either `result` or `n_breaches`.")
        result = basel_traffic_light(n_breaches, n_observations, confidence)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.4))

    # The count axis runs a little past the start of the red zone so the red
    # band is always visible.
    axis_max = max(result.red_min + 2, result.n_breaches + 2)

    # The bands fill the lower part of the strip; the upper part is left clear
    # so the label has its own space and never sits on top of the marker line.
    band_top = 0.62

    # Step 1: draw the three zones as coloured bands.
    # green:  0 .. green_max,  yellow: green_max+1 .. red_min-1,  red: red_min ..
    ax.axvspan(-0.5, result.green_max + 0.5, ymax=band_top,
               color=COLORS["green"], alpha=0.30)
    ax.axvspan(result.green_max + 0.5, result.red_min - 0.5, ymax=band_top,
               color=COLORS["yellow"], alpha=0.30)
    ax.axvspan(result.red_min - 0.5, axis_max + 0.5, ymax=band_top,
               color=COLORS["red"], alpha=0.30)

    # Step 2: mark the observed breach count with a single clean line through
    # the bands (no dot -- the line alone reads as the position).
    zone_color = COLORS[result.zone]
    ax.axvline(result.n_breaches, ymin=0, ymax=band_top,
               color=zone_color, linewidth=2.5, zorder=5)

    # Step 3: the label sits in the clear band ABOVE the line, so text and line
    # never overlap. A small connecting tick keeps it tied to the position.
    ax.annotate(
        f"{result.n_breaches} breaches  –  {result.zone.upper()} zone",
        xy=(result.n_breaches, band_top),
        xytext=(result.n_breaches, 0.82),
        ha="center", va="center", fontsize=10, fontweight="bold",
        color=zone_color,
    )

    ax.set_xlim(-0.5, axis_max + 0.5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    # Keep the title short so it fits a narrow (half-width) column; the zone
    # thresholds go on the x-label, which has the full width to itself.
    # This chart is usually drawn in a narrow half-width column, so the x-label
    # is kept short -- a long single line overhangs both edges of the subplot and
    # can cross the page margin. The full thresholds go on a compact second line.
    style_axes(
        ax,
        title="Basel traffic light",
        xlabel="Number of breaches\n"
               f"n={result.n_observations} · green ≤ {result.green_max} · "
               f"red ≥ {result.red_min}",
    )
    ax.grid(False)
    return ax

"""
Breach timeline -- a strip of breach days, to reveal clustering.

The Christoffersen test asks whether breaches are independent or whether they
arrive in clusters. This chart shows the answer at a glance: each breach is a
vertical tick on a timeline. Evenly spread ticks mean independence; ticks
bunched together mean clustering (a model that fails in stressed periods).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from varlib.plotting._style import COLORS, get_pyplot, style_axes, tidy_x_dates


def breach_timeline(
    breaches: Sequence[float],
    dates: Optional[Sequence[Any]] = None,
    ax=None,
):
    """
    Plot a timeline of breach days as vertical ticks.

    Parameters
    ----------
    breaches
        1-D sequence of 0/1 breach flags in time order.
    dates
        Optional x-axis labels. Defaults to an integer day index.
    ax
        Optional existing Axes. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = get_pyplot()

    flags = np.asarray(breaches, dtype=float)
    x = np.arange(flags.size) if dates is None else np.asarray(dates)
    breach_x = x[flags > 0]

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 1.8))

    # One vertical tick per breach, spanning the full height of the strip.
    ax.vlines(breach_x, ymin=0, ymax=1, color=COLORS["breach"], linewidth=1.4)
    # A faint baseline across the whole period for context.
    ax.axhline(0, color=COLORS["grid"], linewidth=1.0)
    # Anchor the axis to the FULL period, not just the breach span, so the gaps
    # between breaches are shown to scale (that is the whole point of the chart).
    if x.size:
        ax.set_xlim(x[0], x[-1])

    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    style_axes(
        ax,
        title=f"Breach timeline ({int((flags > 0).sum())} breaches)",
        xlabel="Date" if dates is not None else "Day",
    )
    ax.grid(False)
    # This strip is often drawn at half width (next to the traffic light), so
    # cap the date ticks lower than the full-width charts to avoid any overlap.
    tidy_x_dates(ax, dates is not None, max_ticks=4)
    ax.margins(x=0.01)
    return ax

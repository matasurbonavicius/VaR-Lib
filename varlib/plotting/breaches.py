"""
The core backtest chart: VaR forecast versus realised loss, breaches marked.

This is the chart a risk committee actually looks at. For each day it plots:
  * the realised loss (grey bars),
  * the VaR forecast (blue line) -- the level we said losses should rarely cross,
  * the days the loss DID cross it (red markers) -- the breaches.

A well-calibrated model has the right number of red markers, scattered evenly.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from varlib.plotting._style import (
    COLORS,
    add_headroom,
    as_percent,
    get_pyplot,
    style_axes,
    tidy_x_dates,
)


def breaches_chart(
    realised_losses: Sequence[float],
    var_forecasts: Sequence[float],
    dates: Optional[Sequence[Any]] = None,
    confidence: float = 0.99,
    ax=None,
):
    """
    Plot realised losses against the VaR forecast, with breaches marked.

    Parameters
    ----------
    realised_losses
        Realised loss per day (positive = a loss).
    var_forecasts
        VaR forecast per day, aligned with `realised_losses`.
    dates
        Optional x-axis labels (e.g. a pandas DatetimeIndex). Defaults to an
        integer day index.
    confidence
        Confidence level, used only for the legend label.
    ax
        Optional existing Axes to draw on. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = get_pyplot()

    losses = np.asarray(realised_losses, dtype=float)
    forecasts = np.asarray(var_forecasts, dtype=float)
    if losses.shape != forecasts.shape:
        raise ValueError("realised_losses and var_forecasts must be the same length.")

    x = np.arange(losses.size) if dates is None else np.asarray(dates)

    # A breach is a day whose realised loss exceeded the forecast VaR.
    is_breach = losses > forecasts

    if ax is None:
        _, ax = plt.subplots(figsize=(11, 4.5))

    # Realised losses as faint bars in the background.
    ax.bar(x, losses, width=1.0, color=COLORS["loss"], alpha=0.35,
           label="Realised loss")
    # The VaR forecast as a clear line on top.
    ax.plot(x, forecasts, color=COLORS["var"], linewidth=1.6,
            label=f"VaR forecast ({confidence:.0%})")
    # The breaches highlighted as red dots at the realised-loss level.
    ax.scatter(x[is_breach], losses[is_breach], color=COLORS["breach"],
               s=28, zorder=5, label=f"Breaches ({int(is_breach.sum())})")

    style_axes(
        ax,
        title="VaR backtest: forecast vs realised loss",
        xlabel="Date" if dates is not None else "Day",
        ylabel="Loss",
    )
    as_percent(ax, axis="y")
    tidy_x_dates(ax, dates is not None)
    # Headroom so the upper-left legend clears the VaR line and breach markers.
    add_headroom(ax, frac=0.12)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=3)
    ax.margins(x=0.01)
    return ax

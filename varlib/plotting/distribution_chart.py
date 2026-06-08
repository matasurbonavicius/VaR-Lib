"""
Return-distribution chart with VaR and ES lines overlaid.

This shows where in the tail the two risk numbers sit. The histogram is the
distribution of returns; the VaR line marks the loss quantile, and the ES line
sits further out, at the average of the losses beyond VaR. Seeing them on the
distribution makes clear why ES >= VaR and why ES is the more tail-aware number.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from varlib.plotting._style import (
    COLORS,
    add_headroom,
    as_percent,
    get_pyplot,
    style_axes,
)


def distribution_chart(
    returns: Sequence[float],
    var: float,
    expected_shortfall: Optional[float] = None,
    confidence: float = 0.99,
    bins: int = 60,
    ax=None,
):
    """
    Plot the return distribution with VaR (and ES) marked on the loss tail.

    Parameters
    ----------
    returns
        1-D sequence of returns.
    var
        The VaR, as a positive loss fraction. Drawn at return = -var.
    expected_shortfall
        Optional ES, as a positive loss fraction. Drawn at return = -es.
    confidence
        Confidence level, for the labels.
    bins
        Number of histogram bins.
    ax
        Optional existing Axes. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = get_pyplot()

    r = np.asarray(returns, dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    # Step 1: the histogram of returns. The loss tail is on the left.
    counts, edges, patches = ax.hist(
        r, bins=bins, color=COLORS["loss"], alpha=0.45, edgecolor="white", linewidth=0.4
    )

    # Step 2: shade the part of the tail beyond the VaR (returns below -var).
    var_return = -var
    for patch, left in zip(patches, edges[:-1]):
        if left < var_return:
            patch.set_facecolor(COLORS["breach"])
            patch.set_alpha(0.5)

    # Step 3: the VaR line, on the loss side.
    ax.axvline(var_return, color=COLORS["var"], linewidth=2.0,
               label=f"VaR {confidence:.0%} = {var * 100:.2f}%")

    # Step 4: the ES line, further into the tail.
    if expected_shortfall is not None:
        ax.axvline(-expected_shortfall, color=COLORS["es"], linewidth=2.0,
                   linestyle="--", label=f"ES = {expected_shortfall * 100:.2f}%")

    style_axes(
        ax,
        title="Return distribution with VaR and ES",
        xlabel="Return",
        ylabel="Frequency",
    )
    as_percent(ax, axis="x")
    # Reserve space at the top so the title, the legend and the tops of the
    # VaR/ES lines all clear the tallest bar instead of colliding with it.
    add_headroom(ax, frac=0.24)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    return ax

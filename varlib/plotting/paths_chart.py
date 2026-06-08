"""
Simulated-paths chart: the Monte-Carlo walk behind a simulation VaR.

The simulation models (jump-diffusion, historical bootstrap) don't read a
quantile off the data -- they *build* an h-day loss distribution by generating
many price paths and looking at where they end up. This chart makes that visible:
it draws the cumulative-return paths over the holding period, then attaches the
distribution of their terminal returns -- rotated 90 degrees and flush against
the paths' right edge -- so the histogram reads as the pile-up of where the paths
landed, with the VaR marked on the loss tail.

Like every chart here it takes plain data (a `(n_paths, horizon)` matrix of daily
returns and the VaR) and returns the Axes, so it composes and embeds freely. The
caller does the simulation -- one model's paths per call.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from varlib.plotting._style import COLORS, get_pyplot, style_axes


def paths_chart(
    daily_returns,
    var: float,
    confidence: float = 0.99,
    title: Optional[str] = None,
    color: str = COLORS["var"],
    n_shown: Optional[int] = None,
    bins: int = 80,
    ax=None,
):
    """
    Plot simulated cumulative-return paths and their terminal-return distribution.

    Parameters
    ----------
    daily_returns
        A ``(n_paths, horizon)`` array of simulated daily returns -- exactly what
        a simulation model draws internally. Each row is one path; the columns are
        the consecutive daily returns that compound along it.
    var
        The VaR, as a positive loss fraction. Marked at return = -var on the tail.
    confidence
        Confidence level, for the label.
    title
        Optional title (e.g. the model name).
    color
        Path and histogram colour.
    n_shown
        How many paths to actually draw. Defaults to all of them; lower it only if
        rendering every line is too slow. The distribution always uses every path.
    bins
        Number of histogram bins for the terminal distribution.
    ax
        Optional existing Axes for the paths. A new figure is created if omitted.
        The attached histogram is split off the right of this Axes.

    Returns
    -------
    matplotlib.axes.Axes
        The paths Axes. (The histogram is a second Axes glued to its right edge.)
    """
    plt = get_pyplot()

    daily = np.asarray(daily_returns, dtype=float)
    n_paths, horizon = daily.shape
    n_shown = n_paths if n_shown is None else min(int(n_shown), n_paths)

    # Cumulative log return along each path: a random walk starting at 0, in %.
    cum = np.cumsum(daily, axis=1)
    cum = np.hstack([np.zeros((n_paths, 1)), cum])     # every path starts at 0
    pct = 100.0 * cum
    days = np.arange(horizon + 1)
    terminal = pct[:, -1]                              # h-day return (%) of each path
    var_pct = -100.0 * var

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    # One return axis for the row, pinned to the full path range with a hair of
    # margin, so the histogram pile-up lines up 1:1 with where the paths end.
    lo, hi = pct.min(), pct.max()
    pad = 0.02 * (hi - lo)
    ylim = (lo - pad, hi + pad)

    # -- the paths. Faint per-line alpha turns 10k overlapping lines into a
    # density cloud: dark where paths cluster, fading into the rare-path tails.
    alpha = max(0.01, min(0.5, 12.0 / n_shown))
    ax.plot(days, pct[:n_shown].T, color=color, alpha=alpha, lw=0.5)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axhline(var_pct, color=COLORS["breach"], lw=1.4, ls="--")
    ax.set_xlim(0, horizon)
    ax.set_ylim(*ylim)
    ax.margins(y=0)
    style_axes(ax, title=title, xlabel="day", ylabel="cumulative return (%)")
    shown = "all drawn" if n_shown >= n_paths else f"{n_shown:,} drawn"
    ax.text(0.02, 0.05, f"{n_paths:,} paths ({shown})",
            transform=ax.transAxes, fontsize=8, color="0.4")

    # -- the terminal distribution, rotated and glued to the paths' right edge.
    # Split a narrow Axes off the right of `ax`, sharing its (return) y-axis;
    # bars grow rightward, away from the paths, so the pile-up blooms out of
    # exactly where the paths land. Binning over the same ylim makes the top and
    # bottom bars touch the highest and lowest path ends precisely.
    divider = ax.inset_axes([1.0, 0.0, 0.22, 1.0], sharey=ax)
    divider.hist(terminal, bins=np.linspace(ylim[0], ylim[1], bins + 1),
                 orientation="horizontal", color=color, alpha=0.75)
    divider.axhline(var_pct, color=COLORS["breach"], lw=1.6,
                    label=f"VaR {confidence:.0%} = {var * 100:.2f}%")
    divider.set_ylim(*ylim)
    divider.margins(y=0)
    divider.set_xticks([])
    divider.tick_params(labelleft=False)
    for side in ("top", "right", "bottom"):
        divider.spines[side].set_visible(False)
    divider.legend(loc="lower right", frameon=False, fontsize=8)

    return ax

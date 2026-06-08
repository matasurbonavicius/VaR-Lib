"""
Dynamic Quantile chart -- where breach predictability comes from.

The Dynamic Quantile (DQ) test asks whether breaches can be *predicted* from
their own recent history or from the VaR level itself (see
``varlib.backtest.dynamic_quantile``). The single DQ statistic answers "yes/no",
but it hides *which* term is doing the talking: is it a lagged breach (breaches
cluster in time) or the contemporaneous VaR (breaches track the level)?

This chart opens that up. It draws one horizontal bar per regressor in the DQ
regression -- the constant, each lagged hit, and (when it was included) the VaR
level -- sized by that term's contribution to the statistic. A reference line
marks the threshold above which a single term is, on its own, large enough to
matter; bars past it are coloured as breaches, the rest stay grey. So a model
that passes reads as a row of short grey bars, while a rejected model shows
exactly which mechanism gave it away.
"""

from __future__ import annotations

import numpy as np

from varlib.backtest.dynamic_quantile import DynamicQuantileResult
from varlib.plotting._style import COLORS, add_headroom, get_pyplot, style_axes


def dq_chart(
    result: DynamicQuantileResult,
    ax=None,
):
    """
    Draw the per-regressor contributions of the Dynamic Quantile test.

    Each DQ regressor (the constant, the lagged hits, and the VaR level when used)
    gets a bar sized by its share of the DQ statistic. Bars whose single-term
    chi-square contribution clears the 5% one-degree-of-freedom critical value
    (~3.84) are highlighted -- those are the terms that make breaches predictable.

    Parameters
    ----------
    result
        A ``DynamicQuantileResult`` from ``dynamic_quantile_test``.
    ax
        Optional existing Axes. A new figure is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
    """
    plt = get_pyplot()

    beta = result.steps.get("coefficients")
    if beta is None:
        raise ValueError(
            "This DynamicQuantileResult has no stored coefficients to plot."
        )
    beta = np.asarray(beta, dtype=float)
    if beta.size == 0:
        raise ValueError("The DQ coefficient vector is empty; nothing to plot.")

    n_lags = result.n_lags
    includes_var = bool(result.steps.get("includes_var_level", False))

    # Label each regressor in the order the test built them: constant, then the
    # lagged hits Hit_{t-1}..Hit_{t-lags}, then (optionally) the VaR level.
    labels = ["constant"]
    labels += [f"lag {i}" for i in range(1, n_lags + 1)]
    if includes_var:
        labels.append("VaR level")
    # Guard against any mismatch between the labels we expect and the stored
    # coefficient vector -- fall back to generic names rather than crash.
    if len(labels) != beta.size:
        labels = [f"b{i}" for i in range(beta.size)]

    # Per-term contribution to the DQ statistic. The full statistic is
    #   DQ = beta' (X'X) beta / (q(1-q));
    # a clean, interpretable per-term split is the diagonal approximation
    #   c_j = beta_j^2 * (X'X)_jj / (q(1-q)),
    # i.e. how large the statistic would be if only term j moved. Each c_j is on a
    # chi-square(1) scale, so the usual 3.84 critical value reads as "this single
    # term is significant at 5%". (The terms need not sum to DQ when the
    # regressors are correlated, but their relative sizes show where the signal
    # sits.)
    xtx = result.steps.get("xtx")
    q = result.steps.get("expected_rate")
    on_chi2_scale = xtx is not None and q is not None
    if on_chi2_scale:
        diag = np.diag(np.asarray(xtx, dtype=float))
        contrib = (beta ** 2) * diag / (q * (1.0 - q))
    else:
        # xtx was not stored (an older result): fall back to |beta|, which still
        # ranks the terms even if it is not on a chi-square scale.
        contrib = np.abs(beta)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2.4))

    y = np.arange(len(labels))
    # A single term is "significant" once its chi-square(1) contribution clears
    # the 5% critical value -- only meaningful when the bars are on that scale.
    threshold = 3.841 if on_chi2_scale else None
    if threshold is not None:
        colours = [COLORS["breach"] if c >= threshold else COLORS["loss"]
                   for c in contrib]
    else:
        colours = [COLORS["loss"]] * len(labels)

    ax.barh(y, contrib, color=colours, alpha=0.85, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()  # constant on top: reads top-to-bottom in build order

    if threshold is not None:
        ax.axvline(threshold, color="#333333", linestyle="--", linewidth=1.0,
                   zorder=3)
        ax.annotate(
            "5% significance",
            xy=(threshold, len(labels) - 0.5), xytext=(4, 0),
            textcoords="offset points", fontsize=7.5, color="#333333",
            va="bottom", ha="left", rotation=90,
        )

    verdict = "REJECT" if result.reject else "model OK"
    style_axes(
        ax,
        title="Dynamic Quantile -- where predictability sits",
        xlabel=f"per-term contribution to DQ  ·  DQ={result.statistic:.2f} · "
               f"p={result.p_value:.3f} · {verdict}",
    )
    # Horizontal bars: the grid is most useful along x, and a touch of headroom
    # keeps the longest bar (and the threshold label) clear of the right edge.
    ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.grid(False, axis="y")
    add_headroom(ax, frac=0.12, axis="x")
    return ax

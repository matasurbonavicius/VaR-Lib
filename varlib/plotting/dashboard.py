"""
A one-figure backtest dashboard combining all four charts.

This is the convenience entry point: hand it the rolled backtest data and it
lays out the breach chart, the breach timeline, the traffic light, and the
return distribution on a single figure, ready to drop into a report.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from varlib.backtest.traffic_light import basel_traffic_light, count_breaches
from varlib.plotting._style import (
    enforce_horizontal_margins,
    get_pyplot,
    shrink_text_to_width,
)
from varlib.plotting.breaches import breaches_chart
from varlib.plotting.breach_timeline import breach_timeline
from varlib.plotting.traffic_light_chart import traffic_light_chart
from varlib.plotting.distribution_chart import distribution_chart


# A4 portrait, in inches (210 x 297 mm). Used for the print-ready report so the
# PNG and the PDF share one page geometry.
A4_PORTRAIT = (8.27, 11.69)


def backtest_dashboard(
    realised_losses: Sequence[float],
    var_forecasts: Sequence[float],
    expected_shortfall: Optional[float] = None,
    dates: Optional[Sequence[Any]] = None,
    confidence: float = 0.99,
    title: str = "VaR backtest dashboard",
    subtitle: Optional[str] = None,
    footer: Optional[str] = None,
    a4: bool = False,
):
    """
    Build a single figure with all four backtest charts.

    Parameters
    ----------
    realised_losses, var_forecasts
        The rolled backtest series (positive losses, aligned VaR forecasts).
    expected_shortfall
        Optional ES (positive loss fraction) for the distribution chart.
    dates
        Optional x-axis labels for the time-series charts.
    confidence
        Confidence level used throughout.
    title
        Figure title (the report heading).
    subtitle
        Optional second header line under the title, e.g. a run timestamp or a
        one-line description. Drawn smaller and in grey.
    footer
        Optional metadata block printed in the bottom margin. Either:
          * a plain string (newlines honoured), or
          * a sequence of ``(category, text)`` pairs, rendered as a clean
            labelled grid -- the category in a bold left gutter, the value
            beside it -- so the run parameters read as structured fields rather
            than a run-on sentence. This is the report-ready form.
    a4
        When True, size the figure to A4 portrait with print margins and reserve
        bands at the top (title/subtitle) and bottom (footer) so the result is a
        clean, report-ready page for PDF or PNG. When False, the original
        landscape on-screen layout is used.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = get_pyplot()

    losses = np.asarray(realised_losses, dtype=float)
    forecasts = np.asarray(var_forecasts, dtype=float)

    # Derive the breach sequence and traffic-light zone once, up front.
    summary = count_breaches(losses, forecasts)
    breach_flags = np.asarray(summary.steps["is_breach"], dtype=float)
    light = basel_traffic_light(summary.n_breaches, summary.n_observations, confidence)

    if a4:
        # A4 page laid out on a single, consistent margin grid so the whole
        # document reads as one block: the header band, the three chart rows and
        # the footer band all share the same left/right edges. The vertical
        # spacing is deliberately modest -- a report should look full and even,
        # not have cavernous gaps between rows.
        #
        #   page margin (in figure fraction) is identical on the left and right
        #   so the page is visually centred; the header rule, the footer rule and
        #   the title are drawn at this edge to frame the content.
        PAGE_L, PAGE_R = 0.060, 0.060
        left, right = PAGE_L, 1.0 - PAGE_R

        # The AXES box starts inset from the page margin to leave room for the
        # decorations drawn outside it (tick labels, the rotated y-titles). These
        # are only a STARTING point: after everything is drawn, a final pass
        # (enforce_horizontal_margins) measures what was actually rendered and
        # squeezes the axes so no label can cross the page margin, whatever its
        # width. The left gap is wider (rotated title + percentage ticks); the
        # right gap only needs to clear the last x-tick label.
        axes_left = left + 0.055
        axes_right = right - 0.040

        fig = plt.figure(figsize=A4_PORTRAIT)
        # Vertical bands, top-to-bottom (figure fraction):
        #   header   : title + subtitle + a rule beneath them
        #   charts   : the gridspec, between `chart_top` and `chart_bottom`
        #   footer   : a rule + the metadata block, inset from the page edge
        footer_bottom = 0.030 if footer else 0.0
        # The charts stop higher when there is a footer, freeing a taller band at
        # the bottom for a structured, well-spaced metadata block.
        chart_bottom = 0.155 if footer else 0.065
        chart_top = 0.855
        gs = fig.add_gridspec(
            3, 2,
            # The breaches chart (row 0) is the tallest, but it does not need to
            # dominate -- trimming it a little evens the page and gives the footer
            # its room without squashing the lower charts.
            height_ratios=[1.7, 1.05, 1.55],
            hspace=0.55, wspace=0.26,
            left=axes_left, right=axes_right, top=chart_top, bottom=chart_bottom,
        )
        title_y, title_size, subtitle_size = 0.955, 14, 9.5
    else:
        # Original landscape layout for on-screen viewing.
        fig = plt.figure(figsize=(12, 11))
        footer_bottom = 0.02
        gs = fig.add_gridspec(
            3, 2,
            height_ratios=[2.0, 1.15, 1.7],
            hspace=0.75, wspace=0.18,
            left=0.07, right=0.97, top=0.93, bottom=0.07,
        )
        title_y, title_size, subtitle_size = 0.975, 15, 10

    ax_breaches = fig.add_subplot(gs[0, :])
    breaches_chart(losses, forecasts, dates, confidence, ax=ax_breaches)

    ax_timeline = fig.add_subplot(gs[1, 0])
    breach_timeline(breach_flags, dates, ax=ax_timeline)

    ax_light = fig.add_subplot(gs[1, 1])
    traffic_light_chart(result=light, ax=ax_light)

    # The distribution chart wants returns (signed), not losses. Compute the
    # VaR and ES of the realised-loss sample together so the two lines are
    # consistent (ES >= VaR). If an ES was passed in it overrides the derived
    # one, but we still guard against an inconsistent (ES < VaR) value.
    ax_dist = fig.add_subplot(gs[2, :])
    var_level = float(np.quantile(losses, confidence))
    tail = losses[losses >= var_level]
    derived_es = float(tail.mean()) if tail.size else var_level
    es_to_plot = expected_shortfall if expected_shortfall is not None else derived_es
    if es_to_plot < var_level:
        # An ES below VaR cannot be drawn coherently on this sample; fall back
        # to the sample-consistent value rather than show crossed lines.
        es_to_plot = derived_es
    distribution_chart(-losses, var_level, es_to_plot, confidence, ax=ax_dist)
    # Trim the far right tail: returns rarely reach the extreme positive bins, so
    # without this the panel wastes a third of its width on empty space. Clip to
    # a symmetric-ish window around the bulk while always keeping the VaR/ES lines
    # (which sit on the left) comfortably in view.
    r_signed = -losses
    hi = float(np.quantile(r_signed, 0.995))
    lo = min(-es_to_plot, float(np.quantile(r_signed, 0.005)))
    pad = 0.08 * (hi - lo) if hi > lo else 0.01
    ax_dist.set_xlim(lo - pad, hi + pad)

    # ---- Header: title, an optional grey subtitle, and a framing rule. -------
    # The title, the rules and the footer sit at the PAGE margin (the true
    # content edge), while the chart axes are inset further so their tick labels
    # and y-titles land inside that margin rather than crossing it. On the A4
    # page these are the PAGE_L/PAGE_R edges; the landscape layout has no page
    # margin distinct from its axes, so it uses the gridspec edges.
    page_left = PAGE_L if a4 else gs.left
    page_right = (1.0 - PAGE_R) if a4 else gs.right
    fig.suptitle(title, fontsize=title_size, fontweight="bold", x=page_left,
                 y=title_y, ha="left")
    if subtitle:
        fig.text(page_left, title_y - 0.026, subtitle, fontsize=subtitle_size,
                 color="#555555", ha="left", va="top")
    # A rule under the header ties the title block to the page and gives the
    # report a clean masthead. Only drawn for the print-ready A4 page.
    if a4:
        rule_y = title_y - (0.060 if subtitle else 0.030)
        fig.add_artist(
            plt.Line2D([page_left, page_right], [rule_y, rule_y],
                       color="#333333", linewidth=1.1,
                       transform=fig.transFigure)
        )

    # ---- Footer: a rule, then a clean labelled metadata grid beneath it. -----
    # `footer` is either a plain string (one block of text) or a list of
    # (category, value) pairs drawn as a two-column grid: a bold category gutter
    # on the left and the values aligned in a column beside it. Either way the
    # grey rule sits ABOVE the whole block -- never through the text -- with a
    # clear gap, and the lines flow downward from just under the rule so the
    # spacing is predictable regardless of how many rows there are.
    if footer:
        rows = list(footer) if not isinstance(footer, str) else None

        FS = 7.0                  # footer font size (points)
        line_h = 0.0150           # vertical step between rows (figure fraction)
        gap_below_rule = 0.011    # clear space between the rule and the first row
        n_rows = len(rows) if rows else footer.count("\n") + 1

        # The rule sits above the first row; the block grows downward from there.
        first_row_y = footer_bottom + (n_rows - 1) * line_h
        rule_y = first_row_y + gap_below_rule
        fig.add_artist(
            plt.Line2D([page_left, page_right], [rule_y, rule_y],
                       color="#bbbbbb", linewidth=0.8,
                       transform=fig.transFigure)
        )

        if rows:
            # Two columns: a bold category label, then the value. The value
            # column starts at a fixed offset so every value lines up vertically,
            # giving the block a clean tabular structure.
            value_x = page_left + 0.085
            value_texts = []
            for i, (category, value) in enumerate(rows):
                y = first_row_y - i * line_h
                fig.text(page_left, y, category.upper(), fontsize=FS - 0.5,
                         color="#222222", ha="left", va="top",
                         family="monospace", fontweight="bold")
                value_texts.append(
                    fig.text(value_x, y, value, fontsize=FS, color="#555555",
                             ha="left", va="top", family="monospace")
                )
            # The squeezing pass moves axes, not free text; a long value could
            # still overflow, so shrink the value column to fit the right margin.
            shrink_text_to_width(fig, value_texts, right=page_right)
        else:
            fig.text(page_left, first_row_y, footer, fontsize=FS,
                     color="#555555", ha="left", va="top",
                     family="monospace", linespacing=1.6)

    # Final guarantee: measure every chart's actual rendered extent (titles, tick
    # labels, y-titles included) and squeeze the axes so nothing crosses the page
    # margin. This also re-centres the content band, so the left and right
    # margins read as equal rather than the page looking lopsided. The header and
    # footer artists are drawn at the page margin already, so they need no move.
    if a4:
        enforce_horizontal_margins(fig, left=page_left, right=page_right)

    return fig

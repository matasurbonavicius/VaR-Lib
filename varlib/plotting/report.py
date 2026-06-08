"""
Report composer -- selective, multi-page backtest reports.

`backtest_dashboard` lays out a fixed one-page set of charts and is intentionally
left untouched. This module is the *composer* on top of it: you choose which
blocks to render, and if more are asked for than fit one page legibly, the report
is spread across two A4 pages. It does not re-implement any chart -- it calls the
existing primitives (`breaches_chart`, `breach_timeline`, `traffic_light_chart`,
`distribution_chart`, `backtest_panel`) and reuses the header / footer / margin
helpers, so every page matches the dashboard's look with no duplicated layout.

Two public functions:

* ``build_report(...)`` -- returns one Figure, or a list of Figures when the
  requested sections span more than one page.
* ``save_report(fig_or_figs, path)`` -- writes a single Figure straight to the
  path, or a list of Figures as a multi-page PDF (``PdfPages``); for PNG it writes
  ``name_p1.png``, ``name_p2.png``, ... one per page.

Sections
--------
``"breaches"``   forecast vs realised loss, breaches marked (full width)
``"timeline"``   breach-day timeline (half width)
``"traffic_light"`` Basel zones (half width)
``"distribution"``  return histogram with VaR/ES (full width)
``"dq"``         Dynamic Quantile per-term contributions (full width)
``"tests"``      the backtest-verdicts panel (full width)

The default (``sections=None``) reproduces the dashboard's one-page set so
nothing regresses.
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
from varlib.plotting.backtest_panel import backtest_panel
from varlib.plotting.breaches import breaches_chart
from varlib.plotting.breach_timeline import breach_timeline
from varlib.plotting.dashboard import A4_PORTRAIT
from varlib.plotting.distribution_chart import distribution_chart
from varlib.plotting.dq_chart import dq_chart
from varlib.plotting.traffic_light_chart import traffic_light_chart

# Every section the report knows how to draw, in their natural reading order.
ALL_SECTIONS = ("breaches", "timeline", "traffic_light", "distribution", "dq",
                "tests")

# The dashboard's one-page set, used when the caller passes no explicit sections.
DEFAULT_SECTIONS = ("breaches", "timeline", "traffic_light", "distribution")

# A "full"-width section occupies a whole gridspec row; a "half"-width one shares
# its row with the next half section. This drives both layout and page packing.
_WIDTH = {
    "breaches": "full",
    "distribution": "full",
    "dq": "full",
    "tests": "full",
    "timeline": "half",
    "traffic_light": "half",
}

# How many grid ROWS one page holds before we spill onto a second page. A full
# section is one row; a pair of halves shares one row. Three rows fills an A4 page
# legibly alongside the header and footer bands -- the dashboard's own one-page
# set is exactly three rows (breaches / timeline+light / distribution). Asking for
# more (e.g. "all", which adds the tests panel) spills cleanly onto page two.
_ROWS_PER_PAGE = 3


def build_report(
    realised_losses: Sequence[float],
    var_forecasts: Sequence[float],
    *,
    sections: Optional[Sequence[str]] = None,
    expected_shortfall: Optional[float] = None,
    dates: Optional[Sequence[Any]] = None,
    confidence: float = 0.99,
    title: str = "VaR backtest report",
    subtitle: Optional[str] = None,
    footer: Optional[Any] = None,
    backtests: Optional[dict] = None,
):
    """
    Build a one- or multi-page backtest report from a chosen set of sections.

    Parameters
    ----------
    realised_losses, var_forecasts
        The rolled backtest series (positive losses, aligned VaR forecasts).
    sections
        Which blocks to render, any of ``ALL_SECTIONS``. ``None`` (default) uses
        the dashboard's one-page set. ``"all"`` (the string) expands to every
        section. Order is normalised to the natural reading order.
    expected_shortfall, dates, confidence, title, subtitle, footer
        As in ``backtest_dashboard``. The header and footer are drawn on the
        first page; continuation pages get a light "(continued)" header.
    backtests
        Optional dict of pre-computed result objects for the ``"tests"`` panel,
        with any of the keys ``"kupiec"``, ``"christoffersen"``,
        ``"dynamic_quantile"``, ``"traffic_light"``. If ``"tests"`` is requested
        but this is omitted, the panel is skipped (the library does not silently
        invent tests it was not handed).

    Returns
    -------
    matplotlib.figure.Figure | list[matplotlib.figure.Figure]
        One figure if everything fits a page, else a list of figures (one per
        page). ``save_report`` accepts either form.
    """
    losses = np.asarray(realised_losses, dtype=float)
    forecasts = np.asarray(var_forecasts, dtype=float)

    requested = _resolve_sections(sections)

    # Pack the requested sections into pages by their grid-row cost (halves pair
    # up, fulls take a row), capped at _ROWS_PER_PAGE rows per page.
    pages = _paginate(requested)

    # Shared data the section renderers draw from, computed once.
    ctx = _build_context(losses, forecasts, expected_shortfall, dates, confidence,
                          backtests)

    figs = []
    for page_index, page_sections in enumerate(pages):
        is_first = page_index == 0
        page_title = title if is_first else f"{title}  (continued)"
        page_subtitle = subtitle if is_first else None
        page_footer = footer if is_first else None
        figs.append(
            _render_page(page_sections, ctx, confidence, page_title,
                         page_subtitle, page_footer)
        )

    return figs[0] if len(figs) == 1 else figs


def save_report(fig_or_figs, path: str) -> list[str]:
    """
    Save a report (one Figure or a list of them) to `path`.

    A single figure is written straight to `path` (format from the extension).
    A list is written as a multi-page PDF when `path` ends in ``.pdf``; for any
    other extension each page is written separately as ``name_p1.ext``,
    ``name_p2.ext``, ... Returns the list of paths actually written.
    """
    plt = get_pyplot()

    figs = fig_or_figs if isinstance(fig_or_figs, (list, tuple)) else [fig_or_figs]

    # Single page: one file, name unchanged.
    if len(figs) == 1:
        figs[0].savefig(path, dpi=200)
        return [path]

    # Multi-page PDF: all pages in one document.
    if path.lower().endswith(".pdf"):
        from matplotlib.backends.backend_pdf import PdfPages

        with PdfPages(path) as pdf:
            for fig in figs:
                pdf.savefig(fig, dpi=200)
        return [path]

    # Other formats (PNG, ...): one file per page, suffixed _p1, _p2, ...
    if "." in path:
        stem, ext = path.rsplit(".", 1)
    else:
        stem, ext = path, "png"
    written = []
    for i, fig in enumerate(figs, start=1):
        page_path = f"{stem}_p{i}.{ext}"
        fig.savefig(page_path, dpi=200)
        written.append(page_path)
    return written


# -- internals --------------------------------------------------------------


def _resolve_sections(sections: Optional[Sequence[str]]) -> list[str]:
    """Normalise the requested sections to a validated, reading-order list."""
    if sections is None:
        chosen = set(DEFAULT_SECTIONS)
    elif isinstance(sections, str) and sections.lower() == "all":
        chosen = set(ALL_SECTIONS)
    else:
        chosen = set(sections)
        unknown = chosen - set(ALL_SECTIONS)
        if unknown:
            raise ValueError(
                f"Unknown section(s): {sorted(unknown)}. "
                f"Valid sections are {list(ALL_SECTIONS)} (or 'all')."
            )
    # Keep the canonical reading order regardless of how they were passed in.
    return [s for s in ALL_SECTIONS if s in chosen]


def _paginate(sections: list[str]) -> list[list[str]]:
    """Group sections into pages by grid-row cost (<= _ROWS_PER_PAGE per page).

    A full-width section costs one row. Two consecutive half-width sections share
    one row; a lone half also takes a row. This mirrors how `_render_page` lays
    the gridspec out, so the page is always full but never overflows.
    """
    pages: list[list[str]] = []
    current: list[str] = []
    rows_used = 0
    pending_half = False  # a half section is waiting to be paired on its row

    def row_cost(section: str) -> int:
        nonlocal pending_half
        if _WIDTH[section] == "half":
            if pending_half:
                pending_half = False
                return 0           # pairs with the previous half: same row
            pending_half = True
            return 1
        pending_half = False
        return 1

    for section in sections:
        cost = row_cost(section)
        if rows_used + cost > _ROWS_PER_PAGE and current:
            pages.append(current)
            current = []
            rows_used = 0
            pending_half = False
            cost = row_cost(section)
        current.append(section)
        rows_used += cost

    if current:
        pages.append(current)
    return pages


def _build_context(losses, forecasts, expected_shortfall, dates, confidence,
                   backtests):
    """Pre-compute everything the section renderers need, once."""
    summary = count_breaches(losses, forecasts)
    breach_flags = np.asarray(summary.steps["is_breach"], dtype=float)
    light = basel_traffic_light(summary.n_breaches, summary.n_observations, confidence)

    # The distribution chart wants the VaR/ES of the realised-loss sample so the
    # two lines are mutually consistent (ES >= VaR); an explicit ES overrides it
    # but we still guard against a crossed (ES < VaR) value -- identical to the
    # dashboard's handling.
    var_level = float(np.quantile(losses, confidence))
    tail = losses[losses >= var_level]
    derived_es = float(tail.mean()) if tail.size else var_level
    es_to_plot = expected_shortfall if expected_shortfall is not None else derived_es
    if es_to_plot < var_level:
        es_to_plot = derived_es

    return {
        "losses": losses,
        "forecasts": forecasts,
        "dates": dates,
        "breach_flags": breach_flags,
        "light": light,
        "var_level": var_level,
        "es_to_plot": es_to_plot,
        "backtests": backtests or {},
    }


def _render_page(sections, ctx, confidence, title, subtitle, footer):
    """Render one A4 page holding the given sections.

    This reuses the dashboard's page geometry: equal page margins, inset axes, a
    header band (title + subtitle + rule) and an optional structured footer band,
    with the final margin-enforcement pass so nothing crosses the page edge.
    """
    plt = get_pyplot()

    # Page margins and axes insets, matching the dashboard so the two look like
    # one family of documents.
    PAGE_L, PAGE_R = 0.060, 0.060
    left, right = PAGE_L, 1.0 - PAGE_R
    axes_left = left + 0.055
    axes_right = right - 0.040

    fig = plt.figure(figsize=A4_PORTRAIT)
    footer_bottom = 0.030 if footer else 0.0
    chart_bottom = 0.155 if footer else 0.065
    chart_top = 0.855

    # Translate the section list into grid rows: pair consecutive halves onto one
    # row, give each full its own row. `layout` is a list of rows, each row a list
    # of (section, colspan) entries spanning a 2-column grid.
    layout = _row_layout(sections)
    n_rows = len(layout)

    # Row heights echo the dashboard's proportions where the sections coincide; a
    # generic weight is used otherwise so a custom page still looks balanced.
    height_ratios = [_row_height(row) for row in layout]
    gs = fig.add_gridspec(
        n_rows, 2,
        height_ratios=height_ratios,
        hspace=0.55, wspace=0.26,
        left=axes_left, right=axes_right, top=chart_top, bottom=chart_bottom,
    )

    for r, row in enumerate(layout):
        if len(row) == 1 and row[0][1] == 2:
            ax = fig.add_subplot(gs[r, :])
            _draw_section(row[0][0], ax, ctx, confidence)
        else:
            # Two half-width sections side by side.
            for c, (section, _span) in enumerate(row):
                ax = fig.add_subplot(gs[r, c])
                _draw_section(section, ax, ctx, confidence)

    _draw_header_footer(fig, title, subtitle, footer, PAGE_L, PAGE_R)
    enforce_horizontal_margins(fig, left=PAGE_L, right=1.0 - PAGE_R)
    return fig


def _row_layout(sections):
    """Pack sections into grid rows: consecutive halves pair up, fulls stand alone."""
    rows = []
    pending = None
    for section in sections:
        if _WIDTH[section] == "half":
            if pending is None:
                pending = section
            else:
                rows.append([(pending, 1), (section, 1)])
                pending = None
        else:
            if pending is not None:
                rows.append([(pending, 1)])  # a lone half on its own row
                pending = None
            rows.append([(section, 2)])
    if pending is not None:
        rows.append([(pending, 1)])
    return rows


def _row_height(row):
    """A height weight for a grid row, echoing the dashboard's proportions."""
    weights = {
        "breaches": 1.7,
        "distribution": 1.55,
        "dq": 1.2,
        "tests": 1.3,
        "timeline": 1.05,
        "traffic_light": 1.05,
    }
    # A row's height is the tallest section it contains.
    return max(weights[section] for section, _ in row)


def _draw_section(section, ax, ctx, confidence):
    """Dispatch one section name to its chart primitive on the given axes."""
    if section == "breaches":
        breaches_chart(ctx["losses"], ctx["forecasts"], ctx["dates"], confidence, ax=ax)
    elif section == "timeline":
        breach_timeline(ctx["breach_flags"], ctx["dates"], ax=ax)
    elif section == "traffic_light":
        traffic_light_chart(result=ctx["light"], ax=ax)
    elif section == "distribution":
        var_level, es_to_plot = ctx["var_level"], ctx["es_to_plot"]
        distribution_chart(-ctx["losses"], var_level, es_to_plot, confidence, ax=ax)
        # Trim the empty right tail, exactly as the dashboard does.
        r_signed = -ctx["losses"]
        hi = float(np.quantile(r_signed, 0.995))
        lo = min(-es_to_plot, float(np.quantile(r_signed, 0.005)))
        pad = 0.08 * (hi - lo) if hi > lo else 0.01
        ax.set_xlim(lo - pad, hi + pad)
    elif section == "dq":
        dq = ctx["backtests"].get("dynamic_quantile")
        if dq is not None:
            dq_chart(dq, ax=ax)
        else:
            # No DQ result handed in: leave a quiet placeholder rather than crash,
            # mirroring how the tests panel skips tests it was not given.
            ax.set_axis_off()
            ax.text(0.5, 0.5, "Dynamic Quantile result not provided",
                    ha="center", va="center", fontsize=9, color="#999999",
                    transform=ax.transAxes)
    elif section == "tests":
        bt = ctx["backtests"]
        backtest_panel(
            kupiec=bt.get("kupiec"),
            christoffersen=bt.get("christoffersen"),
            dynamic_quantile=bt.get("dynamic_quantile"),
            traffic_light=bt.get("traffic_light", ctx["light"]),
            ax=ax,
        )


def _draw_header_footer(fig, title, subtitle, footer, PAGE_L, PAGE_R):
    """Draw the title/subtitle/rule header and the optional structured footer.

    This is the same masthead-and-footer treatment the dashboard uses, lifted to
    the report so every page shares the look. (Kept here rather than imported so
    `dashboard.py` stays exactly as it is.)
    """
    plt = get_pyplot()
    page_left = PAGE_L
    page_right = 1.0 - PAGE_R
    title_y, title_size, subtitle_size = 0.955, 14, 9.5

    fig.suptitle(title, fontsize=title_size, fontweight="bold", x=page_left,
                 y=title_y, ha="left")
    if subtitle:
        fig.text(page_left, title_y - 0.026, subtitle, fontsize=subtitle_size,
                 color="#555555", ha="left", va="top")
    rule_y = title_y - (0.060 if subtitle else 0.030)
    fig.add_artist(
        plt.Line2D([page_left, page_right], [rule_y, rule_y],
                   color="#333333", linewidth=1.1, transform=fig.transFigure)
    )

    if not footer:
        return

    rows = list(footer) if not isinstance(footer, str) else None
    footer_bottom = 0.030
    FS = 7.0
    line_h = 0.0150
    gap_below_rule = 0.011
    n_rows = len(rows) if rows else footer.count("\n") + 1
    first_row_y = footer_bottom + (n_rows - 1) * line_h
    frule_y = first_row_y + gap_below_rule
    fig.add_artist(
        plt.Line2D([page_left, page_right], [frule_y, frule_y],
                   color="#bbbbbb", linewidth=0.8, transform=fig.transFigure)
    )
    if rows:
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
        shrink_text_to_width(fig, value_texts, right=page_right)
    else:
        fig.text(page_left, first_row_y, footer, fontsize=FS,
                 color="#555555", ha="left", va="top",
                 family="monospace", linespacing=1.6)

"""
Shared plotting helpers: the matplotlib import and a consistent style.

Keeping the import in one place means every chart shares one colour palette and
one set of axis conventions, so the charts look like a set.
"""

from __future__ import annotations


def get_pyplot():
    """Import matplotlib.pyplot."""
    import matplotlib.pyplot as plt

    return plt


# A small, professional palette used consistently across every chart.
COLORS = {
    "var": "#1f77b4",        # blue   -- the VaR forecast line
    "es": "#9467bd",         # purple -- the ES line
    "loss": "#7f7f7f",       # grey   -- realised losses
    "breach": "#d62728",     # red    -- breaches / exceptions
    "green": "#2ca02c",      # Basel green zone
    "yellow": "#ff7f0e",     # Basel yellow zone
    "red": "#d62728",        # Basel red zone
    "grid": "#dddddd",
}


def style_axes(ax, title=None, xlabel=None, ylabel=None):
    """Apply the shared look to an Axes: light grid, no top/right spines."""
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", loc="left")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def as_percent(ax, axis="y"):
    """Format an axis as percentages (0.02 -> '2.0%')."""
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v * 100:.1f}%")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def tidy_x_dates(ax, has_dates, max_ticks=6):
    """Keep the date x-axis readable, with the tick granularity matched to the span.

    The right tick spacing depends entirely on how much time the data covers: a
    few weeks wants day/week ticks, a year wants months, a few years wants years,
    and decades want every Nth year. The default auto-locator is span-aware in
    principle but, given a short series, still tends to land on just the two or
    three year-boundaries it can find -- leaving a near-empty axis (the "only
    2022 and 2023" problem).

    So we measure the actual span from the axis limits and pick an explicit
    granularity for it -- decade, year, quarter, month, week or day -- aiming for
    a comfortable number of ticks (roughly 4-8) whatever the range. The labels
    are angled and right-aligned so they never collide on a narrow subplot.

    Integer day-index axes are left alone (their labels are short).
    """
    if not has_dates:
        return
    import matplotlib.dates as mdates

    # The axis limits are in matplotlib date numbers (days). Their difference is
    # the span in days, which tells us which calendar unit to tick on.
    x0, x1 = ax.get_xlim()
    span_days = max(x1 - x0, 1e-6)
    span_years = span_days / 365.25
    span_months = span_days / 30.44

    # Choose a locator whose natural step yields a comfortable number of marks
    # (~max_ticks) across the span. Each branch ticks on a calendar boundary
    # (year start, month start, Monday, ...) so the labels fall on round,
    # readable dates. The intervals are sized from the span so a short series is
    # densely-but-not-crowdedly ticked rather than left near-empty.
    # `day_level` spans tick on days/weeks; their labels need the month spelled
    # out (a bare "10" is ambiguous), so they get an explicit "DD Mon" format.
    # Month/year-level spans use ConciseDateFormatter, which is excellent there.
    span_weeks = span_days / 7.0
    day_level = False
    if span_years > 12:
        # Many years: every Nth year, N chosen to keep the count in range.
        step = max(1, int(round(span_years / max_ticks)))
        locator = mdates.YearLocator(base=step)
    elif span_years > 2.2:
        # A handful of years: one tick per year.
        locator = mdates.YearLocator()
    elif span_months > 10:
        # ~1 to ~2 years: quarter boundaries (Jan/Apr/Jul/Oct).
        locator = mdates.MonthLocator(bymonth=(1, 4, 7, 10))
    elif span_months > 5:
        # ~5 to ~10 months: every month.
        locator = mdates.MonthLocator()
    elif span_weeks > 8:
        # ~2 to ~5 months: every Nth Monday so the count stays ~4-8 (monthly
        # here would give only 2-4 ticks and leave the axis looking empty).
        step = max(1, int(round(span_weeks / max_ticks)))
        locator = mdates.WeekdayLocator(byweekday=mdates.MO, interval=step)
        day_level = True
    elif span_days > 21:
        # ~3 weeks to ~2 months: weekly, anchored on Mondays.
        locator = mdates.WeekdayLocator(byweekday=mdates.MO)
        day_level = True
    else:
        # Down to days: an auto day locator, capped so labels never crowd.
        locator = mdates.AutoDateLocator(minticks=3, maxticks=max_ticks)
        day_level = True

    ax.xaxis.set_major_locator(locator)
    if day_level:
        # Self-contained "10 Mar" labels: no reliance on a separate axis offset
        # for the month, which is easy to miss on a rotated, squeezed axis.
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    else:
        # ConciseDateFormatter shows the minimum that disambiguates each tick
        # (e.g. "Mar" within a year, the year only when it changes).
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

    # A small rightward tilt with right-alignment guarantees no overlap even on
    # narrow subplots, while staying easy to read.
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_rotation_mode("anchor")
        label.set_horizontalalignment("right")


def enforce_horizontal_margins(fig, left, right):
    """Guarantee no Axes' rendered content crosses the page's left/right margins.

    The chart titles, the rotated y-axis labels and the tick labels are all drawn
    OUTSIDE the axes box, so even a box that sits inside the margins can have
    decorations that spill past them. Rather than hand-tune per-element insets
    (fragile -- it breaks the moment a label gets wider), this measures what was
    actually drawn and squeezes the axes horizontally until everything fits.

    For each Axes we take its *tight* bounding box (the box including all its
    decorations) in figure coordinates. If the leftmost box starts before `left`
    or the rightmost ends after `right`, we compute the single affine map on the
    x-axis -- a scale and a shift -- that pulls the whole content band inside
    [left, right], then re-position every Axes through that same map so the
    columns stay aligned and the page stays balanced.

    `left`, `right` are figure-fraction coordinates (0..1) of the hard margins.
    """
    axes = [ax for ax in fig.axes if ax.get_visible()]
    if not axes:
        return

    inv = fig.transFigure.inverted()

    def content_band():
        # The tight bbox of each Axes (box + all its decorations), in figure
        # fraction; return the union's left and right edges.
        lo, hi = 1.0, 0.0
        for ax in axes:
            bb = ax.get_tightbbox(fig.canvas.get_renderer())
            (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            lo, hi = min(lo, x0), max(hi, x1)
        return lo, hi

    # A hair of safety so content lands just INSIDE the margin, never exactly on
    # it (rotated, anchor-aligned tick labels can otherwise sit a fraction over).
    pad = 0.004
    target_l, target_r = left + pad, right - pad
    target_w = target_r - target_l

    # One affine remap usually suffices, but rotated/anchored labels shift a
    # little when the axes move, so re-measure and correct until it actually fits
    # (or a few passes have run -- it converges fast).
    for _ in range(4):
        fig.canvas.draw()  # refresh text extents for the current axes positions
        content_left, content_right = content_band()
        content_w = content_right - content_left
        if content_w <= 0:
            return
        # Already inside with room to spare -> nothing to do.
        if content_left >= target_l - 1e-4 and content_right <= target_r + 1e-4:
            break

        scale = min(1.0, target_w / content_w)
        new_content_left = target_l + (target_w - content_w * scale) / 2.0

        def remap(x, _cl=content_left, _ncl=new_content_left, _s=scale):
            return _ncl + (x - _cl) * _s

        for ax in axes:
            pos = ax.get_position()  # the axes BOX, figure fraction
            nx0, nx1 = remap(pos.x0), remap(pos.x1)
            ax.set_position([nx0, pos.y0, nx1 - nx0, pos.height])


def shrink_text_to_width(fig, texts, right):
    """Shrink the given figure-text artists until none crosses the `right` margin.

    The axes-squeezing pass only repositions axes; free-floating text (the footer
    grid) is not moved by it. A long value -- a wide data span, an unusual set of
    command-line arguments -- could still run past the right margin. This caps
    that: it finds the widest of `texts`, and if it overflows, scales every one
    of them down by the same factor so the block shrinks together and stays
    aligned, never spilling past `right`.

    `right` is a figure-fraction x-coordinate; `texts` are Text artists already
    added to the figure.
    """
    texts = [t for t in texts if t is not None]
    if not texts:
        return
    inv = fig.transFigure.inverted()

    for _ in range(6):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        worst_overflow = 1.0  # ratio of available width to needed width, <1 = too wide
        for t in texts:
            bb = t.get_window_extent(renderer)
            (x0, _), (x1, _) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
            avail = right - x0
            needed = x1 - x0
            if needed > 0:
                worst_overflow = min(worst_overflow, avail / needed)
        if worst_overflow >= 0.999:
            return  # everything already fits
        # Shrink all texts by the worst ratio (with a hair of headroom) so the
        # block scales down together and the widest line lands inside the margin.
        factor = max(0.5, worst_overflow * 0.99)
        for t in texts:
            t.set_fontsize(t.get_fontsize() * factor)


def add_headroom(ax, frac=0.15, axis="y"):
    """Grow an axis limit by `frac` so titles, lines and labels clear the data.

    Reserves a margin above the data on the chosen axis. Used so the bold title
    drawn inside the axes never sits on top of the tallest bar, and so vertical
    reference lines have room for their labels.
    """
    if axis == "y":
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + (hi - lo) * frac)
    else:
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo, hi + (hi - lo) * frac)

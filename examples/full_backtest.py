"""
Full backtest: roll a chosen VaR model through AAPL's history, print the
standard backtest statistics, and render a print-ready A4 dashboard.

    python examples/full_backtest.py                       # Historical, 99%, 1-day
    python examples/full_backtest.py --model ou
    python examples/full_backtest.py --model jump --confidence 0.975
    python examples/full_backtest.py --format pdf          # PDF only
    python examples/full_backtest.py --window 500          # longer look-back
    python examples/full_backtest.py --horizon 10          # 10-day (Basel) VaR
    python examples/full_backtest.py --horizon 10 --no-overlap  # independent windows

The VaR is computed directly at the chosen --horizon (not sqrt-of-time scaled),
and the realised loss is the cumulative loss over the same number of days, so the
forecast and the outcome are always on the same holding period.

Writes an A4 page to examples/output/:
    dashboard_<model>.pdf   and/or   dashboard_<model>.png

For the individual charts on their own, see examples/charts/.
"""

import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _common import (  # noqa: E402
    MODEL_LABELS,
    ensure_output_dir,
    load_prices,
    make_model,
    parse_args,
    roll_backtest,
    save_page,
)
from varlib.backtest import (  # noqa: E402
    basel_traffic_light,
    christoffersen_test,
    count_breaches,
    dynamic_quantile_test,
    kupiec_pof_test,
)
from varlib.plotting import backtest_dashboard, build_report, save_report  # noqa: E402


def add_report_args(parser):
    """Report-specific flags layered on top of --model / --confidence."""
    parser.add_argument(
        "--format",
        choices=["pdf", "png", "both"],
        default="both",
        help="Output format for the A4 dashboard (default: both).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=250,
        help="Rolling look-back window in trading days (default: 250).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="VaR holding period in trading days (default: 1). The forecast and "
             "the realised loss are both computed over this many days.",
    )
    parser.add_argument(
        "--overlap",
        dest="overlap",
        action="store_true",
        default=True,
        help="Use overlapping h-day windows (default). Uses all data; the "
             "Christoffersen independence test is biased under overlap.",
    )
    parser.add_argument(
        "--no-overlap",
        dest="overlap",
        action="store_false",
        help="Use non-overlapping h-day windows (independent observations, "
             "~1/horizon as many points).",
    )
    parser.add_argument(
        "--sections",
        default=None,
        help="Comma-separated report sections to render, any of: "
             "breaches, timeline, traffic_light, distribution, dq, tests. Use "
             "'all' for every section (spills onto a second A4 page). Omit for "
             "the default one-page dashboard (today's behaviour).",
    )


def compute_report(losses, forecasts, confidence):
    """Run the four backtests and return their results in one dict."""
    summary = count_breaches(losses, forecasts)
    flags = np.array(summary.steps["is_breach"], dtype=float)
    return {
        "summary": summary,
        "kupiec": kupiec_pof_test(flags, confidence),
        "chris": christoffersen_test(flags, confidence),
        "dq": dynamic_quantile_test(flags, confidence, var_forecasts=forecasts),
        "light": basel_traffic_light(
            summary.n_breaches, summary.n_observations, confidence
        ),
    }


def print_report(rep, confidence, label):
    """Print the Kupiec / Christoffersen / DQ / Basel summary to the console."""
    summary, kupiec, chris, dq, light = (
        rep["summary"], rep["kupiec"], rep["chris"], rep["dq"], rep["light"]
    )
    print("=" * 64)
    print(f"{label} VaR backtest  (confidence = {confidence:.0%})")
    print("=" * 64)
    print(f"Observations    : {summary.n_observations}")
    print(f"Breaches        : {summary.n_breaches} "
          f"(rate {summary.breach_rate * 100:.2f}%, "
          f"expected {(1 - confidence) * 100:.2f}%)")
    print(f"Kupiec POF      : p = {kupiec.p_value:.3f} "
          f"-> {'REJECT' if kupiec.reject_at_5pct else 'OK'}")
    print(f"Christoffersen  : p = {chris.p_value_conditional:.3f} "
          f"-> {'REJECT' if chris.reject_conditional else 'OK'}")
    print(f"Dynamic Quantile: p = {dq.p_value:.3f} "
          f"-> {'REJECT' if dq.reject else 'OK'}")
    print(f"Basel zone      : {light.zone.upper()} "
          f"(green<= {light.green_max}, red>= {light.red_min})")


def build_footer(args, rep, prices, generated_at):
    """The page's metadata block, as labelled (category, value) rows.

    Returns one row per category -- inputs, the headline result, the statistical
    tests, and the run/provenance details -- so the dashboard can lay them out as
    a clean grid (bold category gutter, aligned values) rather than a run-on
    sentence. Within a row, fields are separated by a middot so each value reads
    as a list of distinct facts. A printed page is then self-describing and
    reproducible.
    """
    summary, kupiec, chris, dq, light = (
        rep["summary"], rep["kupiec"], rep["chris"], rep["dq"], rep["light"]
    )
    kupiec_v = "OK" if not kupiec.reject_at_5pct else "REJECT"
    chris_v = "OK" if not chris.reject_conditional else "REJECT"
    dq_v = "OK" if not dq.reject else "REJECT"
    span = f"{prices.index.min().date()} – {prices.index.max().date()}"
    cli = " ".join(sys.argv[1:]) or "(defaults)"
    scheme = "overlapping" if args.overlap else "non-overlapping"

    # The Christoffersen test assumes independent observations; overlapping
    # multi-day windows violate that, so flag the verdict as biased in that case.
    chris_note = ""
    if args.horizon > 1 and args.overlap:
        chris_note = " [biased: overlapping windows]"

    return [
        ("Inputs",
         f"Model {MODEL_LABELS[args.model]}  ·  "
         f"Confidence {args.confidence:.1%}  ·  "
         f"Horizon {args.horizon}d ({scheme})  ·  "
         f"Window {args.window}d  ·  "
         f"Data {span} ({len(prices)} days)"),
        ("Result",
         f"Breaches {summary.n_breaches}/{summary.n_observations}  ·  "
         f"Rate {summary.breach_rate:.2%} (expected {1 - args.confidence:.2%})  ·  "
         f"Basel zone {light.zone.upper()}"),
        ("Tests",
         f"Kupiec POF {kupiec_v} (p={kupiec.p_value:.3f})  ·  "
         f"Christoffersen {chris_v} (p={chris.p_value_conditional:.3f})  ·  "
         f"Dynamic Quantile {dq_v} (p={dq.p_value:.3f}){chris_note}"),
        ("Run",
         f"full_backtest.py {cli}  ·  Generated {generated_at}"),
    ]


def main():
    args = parse_args(__doc__, extra_args=add_report_args)
    label = MODEL_LABELS[args.model]
    generated_at = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    prices = load_prices()
    model = make_model(args.model, args.confidence, args.horizon)
    losses, forecasts, dates = roll_backtest(
        prices, model, window=args.window, overlap=args.overlap
    )

    rep = compute_report(losses, forecasts, args.confidence)
    print_report(rep, args.confidence, label)

    horizon_phrase = (
        "1-day" if args.horizon == 1
        else f"{args.horizon}-day ({'overlapping' if args.overlap else 'non-overlapping'})"
    )
    title = f"AAPL {label} VaR — backtest report"
    subtitle = (f"{horizon_phrase} horizon · rolling {args.window}-day window"
                f" · {args.confidence:.0%} confidence · generated {generated_at}")
    footer = build_footer(args, rep, prices, generated_at)

    formats = ["pdf", "png"] if args.format == "both" else [args.format]

    if args.sections is None:
        # Default: the original one-page dashboard, so nothing regresses.
        fig = backtest_dashboard(
            losses, forecasts, dates=dates, confidence=args.confidence,
            title=title, subtitle=subtitle, footer=footer, a4=True,
        )
        for ext in formats:
            save_page(fig, f"dashboard_{args.model}.{ext}")
        return

    # Selective / multi-page report: compose exactly the requested sections. The
    # tests panel is fed the already-computed result objects.
    sections = "all" if args.sections.strip().lower() == "all" else [
        s.strip() for s in args.sections.split(",") if s.strip()
    ]
    backtests = {
        "kupiec": rep["kupiec"],
        "christoffersen": rep["chris"],
        "dynamic_quantile": rep["dq"],
        "traffic_light": rep["light"],
    }
    report = build_report(
        losses, forecasts, sections=sections, dates=dates,
        confidence=args.confidence, title=title, subtitle=subtitle,
        footer=footer, backtests=backtests,
    )
    for ext in formats:
        out = ensure_output_dir()
        written = save_report(report, os.path.join(out, f"report_{args.model}.{ext}"))
        for path in written:
            print(f"Saved {path}")


if __name__ == "__main__":
    main()

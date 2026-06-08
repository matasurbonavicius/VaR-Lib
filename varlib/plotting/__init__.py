"""
Charts for VaR backtests.

This subpackage is the ONLY part of varlib that needs matplotlib, and it is
imported lazily so the core library stays dependency-free (numpy + pandas).
Install the optional extra to use it:

    pip install "varlib[plot]"

One chart per file, each a small function that takes plain data and returns a
matplotlib Axes, so charts compose and can be saved or embedded freely:

    breaches_chart        VaR forecast vs realised loss, breaches marked.
    breach_timeline       A timeline of breach days, to reveal clustering.
    traffic_light_chart   Basel green/yellow/red zones with the breach count.
    distribution_chart    Return histogram with VaR and ES lines overlaid.
    dq_chart              Dynamic Quantile: which terms make breaches predictable.
    backtest_panel        Every backtest verdict as one clean table.

A convenience function, `backtest_dashboard`, lays the four core charts out on
one figure. For selective and multi-page reports, `build_report` / `save_report`
compose those same primitives across one or two A4 pages.
"""

from varlib.plotting.breaches import breaches_chart
from varlib.plotting.breach_timeline import breach_timeline
from varlib.plotting.traffic_light_chart import traffic_light_chart
from varlib.plotting.distribution_chart import distribution_chart
from varlib.plotting.dq_chart import dq_chart
from varlib.plotting.backtest_panel import backtest_panel
from varlib.plotting.dashboard import backtest_dashboard
from varlib.plotting.report import build_report, save_report

__all__ = [
    "breaches_chart",
    "breach_timeline",
    "traffic_light_chart",
    "distribution_chart",
    "dq_chart",
    "backtest_panel",
    "backtest_dashboard",
    "build_report",
    "save_report",
]

"""
Full backtest in a few lines: roll a model through AAPL's history, run every
test, and write a single print-ready dashboard page.

Writes examples/output/dashboard_historical.png and .pdf.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import ensure_output_dir, load_prices  # noqa: E402
from varlib import HistoricalVar, run_backtest  # noqa: E402


def main():
    prices = load_prices()
    out = ensure_output_dir()

    # One call does the roll and all four tests.
    report = run_backtest(HistoricalVar(confidence=0.99), prices=prices, window=250)
    report.print()

    # One polished page, titled and footed automatically -- just give it a path.
    written = report.save(os.path.join(out, "dashboard_historical.png"))
    written += report.save(os.path.join(out, "dashboard_historical.pdf"))

    print()
    for path in written:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

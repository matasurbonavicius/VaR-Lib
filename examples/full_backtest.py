"""
Full backtest in a few lines: roll a model through AAPL's history, run every
test, and write a single print-ready dashboard page.

Writes examples/output/dashboard_historical.png and .pdf.
"""

import os

import pandas as pd

from varlib import HistoricalVar, run_backtest

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT = os.path.join(HERE, "output")


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()
    os.makedirs(OUTPUT, exist_ok=True)

    # One call does the roll and all four tests.
    report = run_backtest(HistoricalVar(confidence=0.99), prices=prices, window=250)
    report.print()

    # One polished page, titled and footed automatically -- just give it a path.
    written = report.save(os.path.join(OUTPUT, "dashboard_historical.png"))
    written += report.save(os.path.join(OUTPUT, "dashboard_historical.pdf"))

    print()
    for path in written:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

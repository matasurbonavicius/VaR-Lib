"""
Full backtest in a few lines: roll a model through AAPL's history, run every
test, and write a single print-ready dashboard page.

Writes examples/output/dashboard_historical.png and .pdf.
"""

import os

import numpy as np
import pandas as pd

from varlib import HistoricalVar, run_backtest

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT = os.path.join(HERE, "output")


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()
    os.makedirs(OUTPUT, exist_ok=True)

    # Models consume returns; log returns keep the price index (minus day one),
    # so the report can still label each step by date.
    returns = np.log(prices / prices.shift(1)).dropna()

    # One call does the roll and every test.
    report = run_backtest(HistoricalVar(confidence=0.99), returns=returns, window=250)
    report.print()

    # One polished page, titled and footed automatically -- just give it a path.
    written = report.save(os.path.join(OUTPUT, "dashboard_historical.png"))
    written += report.save(os.path.join(OUTPUT, "dashboard_historical.pdf"))

    print()
    for path in written:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()

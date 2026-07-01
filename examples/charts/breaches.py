"""
Single chart: VaR forecast vs realised loss, with breaches marked.

This is the chart a risk committee looks at first. Run it on its own to see
just this plot.

    python examples/charts/breaches.py

Writes examples/output/breaches_historical.png. To try another model, change
the one line that builds it.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from varlib import HistoricalVar
from varlib.backtest import rolling_backtest
from varlib.plotting import breaches_chart

HERE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT = os.path.join(HERE, "output")


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()
    # Models consume returns; log returns keep the price index (minus day one).
    returns = np.log(prices / prices.shift(1)).dropna()
    os.makedirs(OUTPUT, exist_ok=True)

    model = HistoricalVar(confidence=0.99)
    losses, forecasts, dates = rolling_backtest(model, returns=returns, window=250)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    breaches_chart(losses, forecasts, dates, 0.99, ax=ax)
    fig.savefig(os.path.join(OUTPUT, "breaches_historical.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

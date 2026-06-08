"""
Single chart: breach timeline -- a strip of breach days, to reveal clustering.

Evenly spread ticks suggest the breaches are independent; ticks bunched
together suggest the model fails in clusters (stressed periods).

    python examples/charts/timeline.py

Writes examples/output/timeline_historical.png. To try another model, change
the one line that builds it.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from varlib import HistoricalVar
from varlib.backtest import count_breaches, rolling_backtest
from varlib.plotting import breach_timeline

HERE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT = os.path.join(HERE, "output")


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()
    os.makedirs(OUTPUT, exist_ok=True)

    model = HistoricalVar(confidence=0.99)
    losses, forecasts, dates = rolling_backtest(model, prices=prices, window=250)

    # The 0/1 breach flags are a traced intermediate of count_breaches.
    summary = count_breaches(losses, forecasts)
    flags = np.array(summary.steps["is_breach"], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 2.2))
    breach_timeline(flags, dates, ax=ax)
    fig.savefig(os.path.join(OUTPUT, "timeline_historical.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

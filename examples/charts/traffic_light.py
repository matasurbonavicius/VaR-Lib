"""
Single chart: Basel traffic light -- the supervisory view of the breach count.

Green / yellow / red zones are derived from the binomial breach distribution; a
single line marks where this model's breach count falls.

    python examples/charts/traffic_light.py

Writes examples/output/traffic_light_historical.png. To try another model,
change the one line that builds it.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from varlib import HistoricalVar
from varlib.backtest import basel_traffic_light, count_breaches, rolling_backtest
from varlib.plotting import traffic_light_chart

HERE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT = os.path.join(HERE, "output")


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()
    os.makedirs(OUTPUT, exist_ok=True)

    model = HistoricalVar(confidence=0.99)
    losses, forecasts, _ = rolling_backtest(model, prices=prices, window=250)

    summary = count_breaches(losses, forecasts)
    light = basel_traffic_light(summary.n_breaches, summary.n_observations, 0.99)

    fig, ax = plt.subplots(figsize=(8, 2.4))
    traffic_light_chart(result=light, ax=ax)
    fig.savefig(os.path.join(OUTPUT, "traffic_light_historical.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

"""
Single chart: return distribution with the VaR and ES lines overlaid.

Shows where in the loss tail the two risk numbers sit, and why ES >= VaR. This
one is a point-in-time estimate on the most recent two years rather than a roll.

    python examples/charts/distribution.py

Writes examples/output/distribution_historical.png. To try another model,
change the one line that builds it.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from varlib import HistoricalVar
from varlib._returns import to_returns
from varlib.plotting import distribution_chart

HERE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT = os.path.join(HERE, "output")


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()
    os.makedirs(OUTPUT, exist_ok=True)

    # Estimate on the most recent ~two years of returns.
    recent = to_returns(prices.to_numpy())[-500:]
    result = HistoricalVar(confidence=0.99).run(returns=recent)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    distribution_chart(recent, result.value, result.expected_shortfall, 0.99, ax=ax)
    fig.savefig(os.path.join(OUTPUT, "distribution_historical.png"), dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

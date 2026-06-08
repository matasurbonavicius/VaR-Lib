"""
End-to-end example on real data: VaR on AAPL, then validate it.

Run with:  python examples/single_instrument.py

Data: daily adjusted-close prices for AAPL, 2020-2024 (examples/data/AAPL.csv),
committed so the example runs offline.

The script:
  1. Loads the price series.
  2. Computes VaR and ES with every model on the most recent two years.
  3. Backtests the Historical model over the full five-year history -- one call
     to `varlib.run_backtest`, which rolls the model and runs all four tests.

Every printed number is also a traced intermediate via `result.steps` /
`result.explain()` (per model) and a plain field on `report.kupiec`, etc.
"""

import os

import pandas as pd

from varlib import (
    HistoricalVar,
    HistoricalBootstrapVar,
    ParametricBrownianVar,
    ParametricOuVar,
    ParametricJumpVar,
    EwmaVar,
    run_backtest,
)

CONFIDENCE = 0.99
DATA = os.path.join(os.path.dirname(__file__), "data", "AAPL.csv")


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()

    print("=" * 64)
    print(f"AAPL VaR  (confidence = {CONFIDENCE})")
    print(f"Data: {prices.index.min().date()} .. {prices.index.max().date()}"
          f"  ({len(prices)} days)")
    print("=" * 64)

    # ---- VaR and ES from every model, on the most recent two years ---------
    recent = prices.loc["2023-01-01":"2024-12-31"]
    print(f"\nVaR and ES estimated on {recent.index.min().date()} .. "
          f"{recent.index.max().date()} ({len(recent)} days):\n")

    models = {
        "Historical": HistoricalVar(CONFIDENCE),
        "Historical bootstrap": HistoricalBootstrapVar(CONFIDENCE, n_resamples=500),
        "Parametric Brownian": ParametricBrownianVar(CONFIDENCE),
        "Parametric OU": ParametricOuVar(CONFIDENCE),
        "Parametric jump": ParametricJumpVar(CONFIDENCE, n_simulations=20_000),
        "EWMA / RiskMetrics": EwmaVar(CONFIDENCE),
    }
    print(f"  {'Model':24s}  {'VaR':>8s}  {'ES':>8s}")
    for name, model in models.items():
        result = model.run(prices=recent.to_numpy())
        print(f"  {name:24s}  {result.value * 100:7.3f}%  "
              f"{result.expected_shortfall * 100:7.3f}%")

    # ---- Backtest the Historical model over the full five-year history ------
    print()
    report = run_backtest(HistoricalVar(CONFIDENCE), prices=prices, window=250)
    report.print("Backtest: rolling Historical VaR, full 2020-2024 history")


if __name__ == "__main__":
    main()

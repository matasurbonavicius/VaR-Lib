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
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import MODELS, MODEL_LABELS, load_prices  # noqa: E402
from varlib import HistoricalVar, run_backtest  # noqa: E402

CONFIDENCE = 0.99


def main():
    prices = load_prices()

    print("=" * 64)
    print(f"AAPL VaR  (confidence = {CONFIDENCE})")
    print(f"Data: {prices.index.min().date()} .. {prices.index.max().date()}"
          f"  ({len(prices)} days)")
    print("=" * 64)

    # ---- VaR and ES from every model, on the most recent two years ---------
    recent = prices.loc["2023-01-01":"2024-12-31"]
    print(f"\nVaR and ES estimated on {recent.index.min().date()} .. "
          f"{recent.index.max().date()} ({len(recent)} days):\n")
    print(f"  {'Model':24s}  {'VaR':>8s}  {'ES':>8s}")
    for name, make in MODELS.items():
        result = make(CONFIDENCE, 1).run(prices=recent.to_numpy())
        print(f"  {MODEL_LABELS[name]:24s}  {result.value * 100:7.3f}%  "
              f"{result.expected_shortfall * 100:7.3f}%")

    # ---- Backtest the Historical model over the full five-year history ------
    print()
    report = run_backtest(HistoricalVar(CONFIDENCE), prices=prices, window=250)
    report.print("Backtest: rolling Historical VaR, full 2020-2024 history")


if __name__ == "__main__":
    main()

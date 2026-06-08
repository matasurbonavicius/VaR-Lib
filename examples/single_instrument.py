"""
End-to-end example on real data: VaR on AAPL, then validate it.

Run with:  python examples/single_instrument.py

Data: daily adjusted-close prices for AAPL, 2020-2024 (examples/data/AAPL.csv).
The CSV is committed so the example needs no network and no extra dependency
beyond pandas.

The script:
  1. Loads the price series.
  2. Computes VaR with every model, using the most recent two years (2023-2024).
  3. Backtests the Historical model by rolling it through the FULL five-year
     history and running all three backtests on the breach sequence.

Every printed number is also available as a traced intermediate via
`result.steps` / `result.explain()`.
"""

import os

import numpy as np
import pandas as pd

from varlib import (
    HistoricalVar,
    HistoricalBootstrapVar,
    ParametricBrownianVar,
    ParametricOuVar,
    ParametricJumpVar,
    EwmaVar,
)
from varlib._returns import to_returns
from varlib.backtest import (
    count_breaches,
    kupiec_pof_test,
    christoffersen_test,
    dynamic_quantile_test,
    basel_traffic_light,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "AAPL.csv")


def load_prices():
    """Load the AAPL price series, indexed by date."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    return df["AAPL"].dropna()


def main():
    prices = load_prices()
    confidence = 0.99

    print("=" * 64)
    print(f"AAPL VaR  (confidence = {confidence})")
    print(f"Data: {prices.index.min().date()} .. {prices.index.max().date()}"
          f"  ({len(prices)} days)")
    print("=" * 64)

    # ---- VaR estimated on the most recent two years (2023-2024) ------------
    recent = prices.loc["2023-01-01":"2024-12-31"]
    print(f"\nVaR and ES estimated on {recent.index.min().date()} .. "
          f"{recent.index.max().date()} ({len(recent)} days):\n")

    models = {
        "Historical": HistoricalVar(confidence),
        "Historical bootstrap": HistoricalBootstrapVar(confidence, n_resamples=500),
        "Parametric Brownian": ParametricBrownianVar(confidence),
        "Parametric OU": ParametricOuVar(confidence),
        "Parametric jump": ParametricJumpVar(confidence, n_simulations=20_000),
        "EWMA / RiskMetrics": EwmaVar(confidence),
    }
    print(f"  {'Model':24s}  {'VaR':>8s}  {'ES':>8s}")
    for name, model in models.items():
        result = model.run(prices=recent.to_numpy())
        print(f"  {name:24s}  {result.value * 100:7.3f}%  "
              f"{result.expected_shortfall * 100:7.3f}%")

    # ---- Backtest the Historical model on the FULL five-year history -------
    print("\n" + "=" * 64)
    print("Backtest: rolling Historical VaR over the full 2020-2024 history")
    print("=" * 64)

    returns = to_returns(prices.to_numpy())
    window = 250  # one trading year of look-back
    forecasts, realised = [], []
    model = HistoricalVar(confidence)
    for t in range(window, len(returns)):
        var_forecast = model.run(returns=returns[t - window:t]).value
        forecasts.append(var_forecast)
        realised.append(-returns[t])  # realised loss the next day

    forecasts_arr = np.array(forecasts)
    summary = count_breaches(np.array(realised), forecasts_arr)
    breach_flags = np.array(summary.steps["is_breach"], dtype=float)
    kupiec = kupiec_pof_test(breach_flags, confidence)
    chris = christoffersen_test(breach_flags, confidence)
    dq = dynamic_quantile_test(breach_flags, confidence, var_forecasts=forecasts_arr)
    light = basel_traffic_light(summary.n_breaches, summary.n_observations, confidence)

    print(f"Observations    : {summary.n_observations}")
    print(f"Breaches        : {summary.n_breaches} "
          f"(rate {summary.breach_rate * 100:.2f}%, expected {(1 - confidence) * 100:.2f}%)")
    print(f"Kupiec POF      : p = {kupiec.p_value:.3f} "
          f"-> {'REJECT' if kupiec.reject_at_5pct else 'OK'}")
    print(f"Christoffersen  : p = {chris.p_value_conditional:.3f} "
          f"-> {'REJECT' if chris.reject_conditional else 'OK'}")
    print(f"Dynamic Quantile: p = {dq.p_value:.3f} "
          f"-> {'REJECT' if dq.reject else 'OK'}")
    print(f"Basel zone      : {light.zone.upper()} "
          f"(green<= {light.green_max}, red>= {light.red_min})")


if __name__ == "__main__":
    main()

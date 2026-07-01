"""
End-to-end example on real data: every VaR model, estimated *and* graded.

Run with:  python examples/single_instrument.py

Data: daily adjusted-close prices for AAPL, 2020-2024 (examples/data/AAPL.csv),
committed so the example runs offline.

The script builds ONE table. Each row is a model, and each model is:
  1. Estimated -- its VaR and ES today (on the most recent two years).
  2. Graded -- rolled through the full five-year history and run through every
     backtest, so you can see whether that VaR number can be trusted.

WHY BOTH COLUMNS MATTER. The VaR/ES is a single number for today; it tells you
nothing about whether the model is any good. Accuracy is a property of the model
*rolled through history*: did reality breach the forecast about as often as
promised (Kupiec), and were the breaches unpredictable rather than clustered in a
crisis (Dynamic Quantile)? A model can ace one and fail the other, and the two
failures mean opposite things -- so the honest verdict needs both.

  Kupiec POF        right *number* of breaches?      (fail => VaR too low/high)
  Dynamic Quantile  breaches *unpredictable*?         (fail => doesn't adapt)
  Basel zone        supervisor's green/yellow/red     (blunt breach-count check,
                    over the most recent 250 trading days only, per the rule)

VERDICT is PASS only when a model clears all three: Kupiec ok, DQ ok, and green.
Watch for the classic trap -- plain Historical gets the breach *count* right but
its breaches cluster (it can't react to a volatility spike), so it fails DQ. The
volatility-aware models (EWMA, FHS) are the ones that tend to pass both.

Every printed number is also a traced field: per-model via `result.steps` /
`result.explain()`, and per-backtest via `report.kupiec`, `report.dynamic_quantile`, etc.

Takes ~30s: grading rolls all eight models through ~1000 days, and FHS refits a
GARCH model at every step. Everything else is near-instant.
"""

import os
import warnings

import numpy as np
import pandas as pd

from varlib import (
    HistoricalVar,
    AgeWeightedHistoricalVar,
    HistoricalBootstrapVar,
    ParametricBrownianVar,
    ParametricOuVar,
    ParametricJumpVar,
    EwmaVar,
    FilteredHistoricalSimulationVar,
    run_backtest,
)

CONFIDENCE = 0.99
WINDOW = 250
DATA = os.path.join(os.path.dirname(__file__), "data", "AAPL.csv")


def build_models():
    """Every model in the library, keyed by display name."""
    return {
        "Historical": HistoricalVar(CONFIDENCE),
        "Age-weighted historical": AgeWeightedHistoricalVar(CONFIDENCE, lambda_decay=0.98),
        "Historical bootstrap": HistoricalBootstrapVar(CONFIDENCE, n_resamples=500),
        "Parametric Brownian": ParametricBrownianVar(CONFIDENCE),
        "Parametric OU": ParametricOuVar(CONFIDENCE),
        "Parametric jump": ParametricJumpVar(CONFIDENCE, n_simulations=20_000),
        "EWMA / RiskMetrics": EwmaVar(CONFIDENCE),
        "Filtered historical (FHS)": FilteredHistoricalSimulationVar(CONFIDENCE),
    }


def grade(name, model, recent_returns, full_returns):
    """Estimate a model today and grade it over history -> one table row.

    Returns a dict of the cells: the VaR/ES point estimate on `recent_returns`,
    and the breach count / rate / three test verdicts from rolling the model over
    `full_returns`.
    """
    # The number you would quote today: VaR and ES on the most recent window.
    today = model.run(recent_returns)

    # The evidence you would trust it on: roll it through the whole history and
    # run every backtest. Horizon 1 keeps the Dynamic Quantile test unbiased
    # (overlapping multi-day windows are serially dependent, which biases it).
    report = run_backtest(model, returns=full_returns, window=WINDOW)

    kupiec_ok = not report.kupiec.reject_at_5pct
    dq_ok = not report.dynamic_quantile.reject
    green = report.traffic_light.zone == "green"

    return {
        "name": name,
        "var": today.value,
        "es": today.expected_shortfall,
        "n_observations": report.summary.n_observations,
        "n_breaches": report.summary.n_breaches,
        "rate": report.summary.breach_rate,
        "kupiec_p": report.kupiec.p_value,
        "kupiec_ok": kupiec_ok,
        "dq_p": report.dynamic_quantile.p_value,
        "dq_ok": dq_ok,
        "zone": report.traffic_light.zone,
        "verdict": "PASS" if (kupiec_ok and dq_ok and green) else "FAIL",
    }


def print_table(rows, n_obs):
    """One line per model: today's estimate on the left, its grade on the right."""
    ok = lambda flag: "ok " if flag else "REJ"
    header = (
        f"  {'Model':26s}  {'VaR':>7s}  {'ES':>7s}  "
        f"{'Breaches':>8s}  {'Rate':>6s}  "
        f"{'Kupiec':>10s}  {'DQ':>10s}  {'Basel':>7s}  {'Verdict':>7s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in rows:
        print(
            f"  {r['name']:26s}  "
            f"{r['var'] * 100:6.2f}%  {r['es'] * 100:6.2f}%  "
            f"{r['n_breaches']:8d}  {r['rate'] * 100:5.2f}%  "
            f"{ok(r['kupiec_ok'])} p={r['kupiec_p']:.2f}  "
            f"{ok(r['dq_ok'])} p={r['dq_p']:.2f}  "
            f"{r['zone'].upper():>7s}  {r['verdict']:>7s}"
        )
    expected = (1 - CONFIDENCE) * 100
    print(
        f"\n  {n_obs} rolling daily forecasts at {CONFIDENCE:.0%}  "
        f"(expected breach rate {expected:.2f}%).  "
        f"VaR/ES are today's estimate; the rest grade the model over history.\n"
        f"  Kupiec & DQ use the full history; Basel zones only the most recent\n"
        f"  250 trading days (the regulatory window).\n"
        f"  VERDICT = PASS only if Kupiec ok AND DQ ok AND Basel green."
    )


def main():
    # arch's GARCH optimiser can emit convergence chatter on some rolling windows;
    # it is not what this example is about, so quiet it for a clean table.
    warnings.filterwarnings("ignore")

    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()

    print("=" * 96)
    print(f"AAPL VaR -- every model, estimated and graded  (confidence = {CONFIDENCE:.0%})")
    print(f"Data: {prices.index.min().date()} .. {prices.index.max().date()}"
          f"  ({len(prices)} days)")
    print("=" * 96)

    # Today's estimate uses the most recent two years; the grade rolls the model
    # through the full history (log returns keep the price index for dated steps).
    recent = prices.loc["2023-01-01":"2024-12-31"]
    recent_returns = np.diff(np.log(recent.to_numpy()))
    full_returns = np.log(prices / prices.shift(1)).dropna()

    print(f"\nVaR/ES estimated on {recent.index.min().date()} .. "
          f"{recent.index.max().date()} ({len(recent)} days); "
          f"graded on the full history, rolling {WINDOW}-day window.\n")

    rows = [
        grade(name, model, recent_returns, full_returns)
        for name, model in build_models().items()
    ]
    # All models roll over the same series, so they share the observation count.
    n_obs = rows[0]["n_observations"]
    print_table(rows, n_obs)


if __name__ == "__main__":
    main()

# VaR-Lib - Value-at-risk focused lightweight library

*Eight VaR models · ES on every one · three backtests ·
one-call reports.*

A small Value at Risk library built on three ideas:

1. **Readable.** One VaR method per file, each formula written out step by step,
   no black boxes.
2. **Traceable.** Every calculation records each intermediate it produces, so you
   can audit a number line by line (`result.steps` / `result.explain()`).
3. **Validated.** A VaR number is only trustworthy once it's backtested. Industry-standard checks are built in.

---

## In one picture

One call rolls a 99% VaR model through five years of real AAPL prices, runs every
test, and writes this page (`examples/full_backtest.py`):

![Backtest dashboard](examples/output/dashboard_historical.png)

The same data, every model — each **estimated** (its VaR/ES today) *and*
**graded** (rolled through history, run through every backtest), so you can see
which number to trust (`examples/single_instrument.py`):

```
  Model                           VaR       ES  Breaches    Rate      Kupiec          DQ    Basel  Verdict
  --------------------------------------------------------------------------------------------------------
  Historical                    2.97%    4.05%        11   1.09%  ok  p=0.77  REJ p=0.00    GREEN     FAIL
  Age-weighted historical       2.91%    3.46%        12   1.19%  ok  p=0.55  ok  p=0.20    GREEN     PASS
  Historical bootstrap          3.23%    3.91%        11   1.09%  ok  p=0.77  REJ p=0.00    GREEN     FAIL
  Parametric Brownian           3.01%    3.47%        13   1.29%  ok  p=0.37  REJ p=0.00    GREEN     FAIL
  Parametric OU                 3.06%    3.52%        12   1.19%  ok  p=0.55  REJ p=0.00    GREEN     FAIL
  Parametric jump               2.94%    4.51%        13   1.29%  ok  p=0.37  REJ p=0.00    GREEN     FAIL
  EWMA / RiskMetrics            2.31%    2.66%        19   1.89%  REJ p=0.01  REJ p=0.00   YELLOW     FAIL
  Filtered historical (FHS)     3.68%    4.19%        14   1.39%  ok  p=0.24  ok  p=0.11    GREEN     PASS

  1007 rolling daily forecasts at 99%  (expected breach rate 1.00%).
  Kupiec & DQ use the full history; Basel zones only the most recent
  250 trading days (the regulatory window).
  VERDICT = PASS only if Kupiec ok AND DQ ok AND Basel green.
```

The VaR number alone tells you nothing about whether to trust it — accuracy is a
property of the model *rolled through history*. Note the trap on the first row:
plain **Historical** gets the breach *count* right (Kupiec ok) but its breaches
**cluster** in the 2020 crash (DQ rejected), because a static quantile can't react
to a volatility spike. Only the volatility-aware models — **Age-weighted** and
**FHS** — pass all three tests here. And the lowest VaR (**EWMA**, 2.31%) is the
*least* safe: it breaches 19 times and lands in the yellow Basel zone.

## Install

```bash
pip install -e .
```

Pulls numpy, pandas, scipy, matplotlib (charts), and pytest (tests). scipy
supplies the standard statistical functions the library relies on: the Normal
pdf/quantile used by the EWMA model, and the binomial and chi-square
distributions behind the Kupiec, Dynamic Quantile, and Basel traffic-light
backtests. It is a relatively heavy dependency (a large, compiled package),
pulled in for convenience and correctness over reimplementing these functions
by hand.

Filtered Historical Simulation needs a GARCH filter, supplied by the
[`arch`](https://pypi.org/project/arch/) package. It (and its statsmodels
dependency) is heavy, so it is kept optional — install it only if you want FHS:

```bash
pip install -e ".[fhs]"
```

Every other model works without it.

## Quick start

```python
import numpy as np
from varlib import HistoricalVar

prices = np.array([100, 101, 99, 102, 98, 100, 103, 97])
returns = np.diff(np.log(prices))                            # your log returns
result = HistoricalVar(confidence=0.99).run(returns)         # models take returns

result.value                 # the VaR, a positive loss fraction (×position = money)
result.expected_shortfall    # the ES (a.k.a. CVaR), always >= VaR
result.explain()             # full step-by-step trace of both
```

Every model returns the same `VarResult`, with `value`, `expected_shortfall`,
`confidence`, `horizon`, `method`, and `steps` (every intermediate, keyed by
name).

## The whole backtest in one call

`run_backtest` rolls a model through history, runs every test, and returns a
`BacktestReport` that prints, plots, or saves itself:

```python
from varlib import HistoricalVar, run_backtest

report = run_backtest(HistoricalVar(0.99), returns=returns, window=250)

report.print()                 # the console summary (breaches + every test)
report.save("backtest.pdf")    # a single print-ready dashboard page

report.kupiec.p_value          # every result is a plain field, nothing hidden
report.traffic_light.zone
```

The saved page is **print-ready** - the title and a self-describing
metrics footer (inputs · result · test verdicts) are generated from the run, so
you pass data, not styling. That is all the examples do: load data, call
`run_backtest`, print/save.

## The models

Each lives under `varlib/models/`, grouped by how much they assume — from
`non_parametric/` (assume nothing) through `semi_parametric/` (empirical shocks,
parametric dynamics) to `parametric/` (assume a full distribution):

| Model                              | Family          | Assumes                                              |
|------------------------------------|-----------------|------------------------------------------------------|
| `HistoricalVar`                    | non-parametric  | Nothing — empirical quantile of past losses.         |
| `AgeWeightedHistoricalVar`         | non-parametric  | As Historical, but recent days count more (BRW).     |
| `HistoricalBootstrapVar`           | non-parametric  | Future = a reshuffle of the past; gives a std error. |
| `FilteredHistoricalSimulationVar`  | semi-parametric | GARCH filter + bootstrap real shocks at today's vol (BGV). |
| `ParametricBrownianVar`            | parametric      | Returns are Normal (variance-covariance / Gaussian). |
| `ParametricOuVar`                  | parametric      | Returns mean-revert (Ornstein–Uhlenbeck / AR(1)).    |
| `ParametricJumpVar`                | parametric      | Normal diffusion **plus** rare Merton jumps (fat tails). |
| `EwmaVar`                          | parametric      | EWMA volatility (RiskMetrics λ=0.94); reacts to clustering. |

All take `confidence` and `horizon`, run on a `returns` series (compute log
returns from prices yourself, e.g. `np.diff(np.log(prices))`), and
auto-calibrate from the data. The `horizon` is computed **directly** at the
holding period — the h-day VaR is the quantile of the h-day loss distribution,
not a one-day figure scaled by √h (which is plainly wrong for mean-reverting
series, where the variance should saturate).

### Inspecting the internals

The simulation models build the h-day loss distribution by generating thousands of
price paths. Because every intermediate is traced, you can pull those paths out
and inspect. Here are 10,000 21-day paths per model, with the distribution
of where they end up attached on the right
(`examples/charts/paths.py`):

![Simulated paths and the loss tail they form](examples/output/paths.png)

## The backtests

Under `varlib/backtest/`. `run_backtest` runs all three; you can also call them
directly on any `(losses, forecasts)` pair:

| Test                | Function                | Question                                |
|---------------------|-------------------------|-----------------------------------------|
| Kupiec POF          | `kupiec_pof_test`       | The right **number** of breaches?       |
| Dynamic Quantile    | `dynamic_quantile_test` | Breaches **predictable** — clustered or VaR-correlated (Engle-Manganelli)? |
| Basel traffic light | `basel_traffic_light`   | Which supervisory zone (green/yellow/red)? |

## Examples

Self-contained, no arguments - each loads the data and calls the library
directly, so you can read any one top to bottom. To try another model, change the
one line that builds it.

```bash
python examples/single_instrument.py        # every model, estimated + graded (the table above)
python examples/full_backtest.py            # the dashboard page above (PNG + PDF)
python examples/charts/breaches.py          # each chart on its own
python examples/charts/paths.py             # the simulated-paths picture above
```

## Design notes

- **Log returns** throughout
- **Reproducible.** Every model that simulates takes a seed

## Testing

```bash
pytest -q
```

## License

MIT.

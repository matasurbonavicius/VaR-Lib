# varlib - Value-at-risk focused lightweight library

*Six VaR models · Expected Shortfall on every one · three regulatory backtests ·
one-call reports · numpy + pandas only.*

A small Value at Risk library built on three ideas:

1. **Readable.** One VaR method per file, each formula written out step by step —
   no black boxes.
2. **Traceable.** Every calculation records each intermediate it produces, so you
   can audit a number line by line (`result.steps` / `result.explain()`).
3. **Validated.** A VaR number is only trustworthy once it's backtested. The four
   industry-standard checks are built in.

---

## In one picture

One call rolls a 99% VaR model through five years of real AAPL prices, runs every
test, and writes this page (`examples/full_backtest.py`):

![Backtest dashboard](examples/output/dashboard_historical.png)

The same data, every model, on the console (`examples/single_instrument.py`):

```
VaR and ES estimated on 2023-01-03 .. 2024-12-31 (502 days):

  Model                          VaR        ES
  Historical                  2.969%    4.048%
  Historical bootstrap        3.227%    3.908%
  Parametric Brownian         2.980%    3.435%
  Parametric OU               3.034%    3.489%
  Parametric jump             2.936%    4.509%
  EWMA / RiskMetrics          2.308%    2.664%

Backtest: rolling Historical VaR, full 2020-2024 history  (confidence = 99%)
  Observations    : 1007
  Breaches        : 11 (rate 1.09%, expected 1.00%)
  Kupiec POF      : p = 0.772 -> OK
  Dynamic Quantile: p = 0.000 -> REJECT
  Basel zone      : GREEN (green<= 15, red>= 24)
```

The **Dynamic Quantile** test rejects where Kupiec passes: the breach *count* is
fine, but the breaches are still predictable from the VaR level — the dependence
a count-only test can't see. That's the case for testing more than the count.

---

## Install

```bash
pip install -e .
```

Pulls numpy, pandas, matplotlib (charts), and pytest (tests). The VaR engine
itself uses only numpy + pandas.

## Quick start

```python
import numpy as np
from varlib import HistoricalVar

prices = np.array([100, 101, 99, 102, 98, 100, 103, 97])
result = HistoricalVar(confidence=0.99).run(prices=prices)   # or run(returns=...)

result.value                 # the VaR, a positive loss fraction (×position = money)
result.expected_shortfall    # the ES (a.k.a. CVaR), always >= VaR
result.explain()             # full step-by-step trace of both
```

Every model returns the same `VarResult`, with `value`, `expected_shortfall`,
`confidence`, `horizon`, `method`, and `steps` (every intermediate, keyed by
name). ES is reported on **every** model — it sees the whole tail, not one point
on it — computed the way each model's assumptions allow (closed form, simulated
tail mean, or bootstrap).

## The whole backtest in one call

`run_backtest` rolls a model through history, runs every test, and returns a
`BacktestReport` that prints, plots, or saves itself:

```python
from varlib import HistoricalVar, run_backtest

report = run_backtest(HistoricalVar(0.99), prices=prices, window=250)

report.print()                 # the console summary (breaches + every test)
report.save("backtest.pdf")    # a single print-ready dashboard page

report.kupiec.p_value          # every result is a plain field, nothing hidden
report.traffic_light.zone
```

The saved page is **print-ready by default** — the title and a self-describing
metrics footer (inputs · result · test verdicts) are generated from the run, so
you pass data, not styling. That is all the examples do: load data, call
`run_backtest`, print/save.

## The models

Each lives in its own file under `varlib/models/`:

| Model                    | Assumes                                              |
|--------------------------|------------------------------------------------------|
| `HistoricalVar`          | Nothing — empirical quantile of past losses.         |
| `HistoricalBootstrapVar` | Future = a reshuffle of the past; gives a std error. |
| `ParametricBrownianVar`  | Returns are Normal (variance-covariance / Gaussian). |
| `ParametricOuVar`        | Returns mean-revert (Ornstein–Uhlenbeck / AR(1)).    |
| `ParametricJumpVar`      | Normal diffusion **plus** rare Merton jumps (fat tails). |
| `EwmaVar`                | EWMA volatility (RiskMetrics λ=0.94); reacts to clustering. |

All take `confidence` and `horizon`, accept `prices=` or `returns=`, and
auto-calibrate from the data. The `horizon` is computed **directly** at the
holding period — the h-day VaR is the quantile of the h-day loss distribution,
not a one-day figure scaled by √h (which is plainly wrong for mean-reverting
series, where the variance should saturate).

### Inspecting the internals

The simulation models don't read a quantile off the data — they *build* the
h-day loss distribution by generating thousands of price paths. Because every
intermediate is traced, you can pull those paths out and look at them. Here are
10,000 21-day paths per model, with the distribution of where they end up — the
exact loss tail the VaR is read from — attached on the right
(`examples/charts/paths.py`):

![Simulated paths and the loss tail they form](examples/output/paths.png)

The paths fan out, pile up into the terminal distribution, and the VaR is the
crimson line in its loss tail. The jump-diffusion tails (top) reach further than
the bootstrap's (bottom) — the rare Merton jumps at work. Nothing here is a black
box: it's the model's own intermediate data, the same `result.steps` you can
audit on any single run.

## The backtests

Under `varlib/backtest/`. `run_backtest` runs all three; you can also call them
directly on any `(losses, forecasts)` pair:

| Test                | Function                | Question                                |
|---------------------|-------------------------|-----------------------------------------|
| Kupiec POF          | `kupiec_pof_test`       | The right **number** of breaches?       |
| Dynamic Quantile    | `dynamic_quantile_test` | Breaches **predictable** — clustered or VaR-correlated (Engle-Manganelli)? |
| Basel traffic light | `basel_traffic_light`   | Which supervisory zone (green/yellow/red)? |

The building blocks `run_backtest` composes are public too: `rolling_var` /
`rolling_backtest` (series in, aligned forecasts out, each lined up with the loss
realised over the *same* holding period), the `varlib.plotting` charts (one per
file, each returning a matplotlib Axes), and `build_report` / `save_report` for
selective, multi-page PDFs. See the examples.

## Examples

Self-contained, no arguments — each loads the data and calls the library
directly, so you can read any one top to bottom. To try another model, change the
one line that builds it.

```bash
python examples/single_instrument.py        # every model + a backtest, console
python examples/full_backtest.py            # the dashboard page above (PNG + PDF)
python examples/charts/breaches.py          # each chart on its own
python examples/charts/paths.py             # the simulated-paths picture above
```

## Design notes

- **Log returns** throughout — additive over time, so the h-day return is the
  sum of the daily ones, and forecast and realised loss share the same footing.
- **No scipy.** The normal quantile (Acklam) and chi-square tail (incomplete
  gamma) are implemented from standard numerical routines, keeping the footprint
  at numpy + pandas.
- **Reproducible.** Every model that simulates takes a `seed`.

## Testing

```bash
pytest -q
```

Each VaR method has its own test file; helpers (OU calibration, jump separation)
get a sub-folder so they're tested apart from the VaR that uses them. Coverage
hits what matters for a risk library: ES ≥ VaR everywhere, the Gaussian closed
forms against textbook values, the numerical routines against known quantiles, OU
parameter recovery on simulated data, and the Basel zones against the published
250-day table.

## License

MIT.

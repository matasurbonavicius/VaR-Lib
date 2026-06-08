# Examples

Runnable examples on real data — daily AAPL prices, 2020–2024, committed in
`data/AAPL.csv`, so everything runs offline.

Charting needs the optional plotting extra:

```bash
pip install -e ".[plot]"
```

## Choosing a model

Every charting example takes the same two flags:

| Flag           | Values                                          | Default      |
|----------------|-------------------------------------------------|--------------|
| `--model`      | `historical`, `bootstrap`, `brownian`, `ou`, `jump` | `historical` |
| `--confidence` | a level in (0, 1), e.g. `0.99`, `0.975`         | `0.99`       |

Generated PNGs are written to `output/` (created on first run), named by model,
e.g. `output/dashboard_brownian.png`.

## The full backtest

Rolls the chosen model day-by-day through the whole history, prints the Kupiec /
Christoffersen / Basel statistics, and renders the one-figure dashboard:

```bash
python examples/full_backtest.py                       # Historical, 99%
python examples/full_backtest.py --model ou
python examples/full_backtest.py --model jump --confidence 0.975
```

## Individual charts

Each script in `charts/` renders exactly one plot, so you can look at them in
isolation:

| Script                       | Chart                                            |
|------------------------------|--------------------------------------------------|
| `charts/breaches.py`         | VaR forecast vs realised loss, breaches in red.  |
| `charts/timeline.py`         | Timeline of breach days, to reveal clustering.   |
| `charts/traffic_light.py`    | Basel green/yellow/red zones with the count.     |
| `charts/distribution.py`     | Return histogram with the VaR and ES lines.      |

```bash
python examples/charts/breaches.py      --model brownian
python examples/charts/timeline.py      --model ou
python examples/charts/traffic_light.py --model jump
python examples/charts/distribution.py  --model historical
```

## Console-only

`single_instrument.py` is the end-to-end workflow with no charts: estimate VaR
with every model on the last two years, then roll the Historical model through
the full history and print the backtest. Needs only numpy + pandas.

```bash
python examples/single_instrument.py
```

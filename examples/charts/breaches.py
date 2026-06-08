"""
Single chart: VaR forecast vs realised loss, with breaches marked.

This is the chart a risk committee looks at first. Run it on its own to see
just this plot for the model you choose.

    python examples/charts/breaches.py --model historical
    python examples/charts/breaches.py --model brownian

Writes examples/output/breaches_<model>.png
"""

import os
import sys

# Allow running this file directly: make examples/ importable for _common.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import load_prices, make_model, parse_args, roll_backtest, save  # noqa: E402
from varlib.plotting import breaches_chart  # noqa: E402
from varlib.plotting._style import get_pyplot  # noqa: E402


def main():
    args = parse_args(__doc__)
    plt = get_pyplot()

    prices = load_prices()
    model = make_model(args.model, args.confidence)
    losses, forecasts, dates = roll_backtest(prices, model)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    breaches_chart(losses, forecasts, dates, args.confidence, ax=ax)
    save(fig, f"breaches_{args.model}.png")


if __name__ == "__main__":
    main()

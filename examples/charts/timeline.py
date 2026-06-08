"""
Single chart: breach timeline -- a strip of breach days, to reveal clustering.

Evenly spread ticks suggest the breaches are independent; ticks bunched
together suggest the model fails in clusters (stressed periods).

    python examples/charts/timeline.py --model historical

Writes examples/output/timeline_<model>.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from _common import load_prices, make_model, parse_args, roll_backtest, save  # noqa: E402
from varlib.backtest import count_breaches  # noqa: E402
from varlib.plotting import breach_timeline  # noqa: E402
from varlib.plotting._style import get_pyplot  # noqa: E402


def main():
    args = parse_args(__doc__)
    plt = get_pyplot()

    prices = load_prices()
    model = make_model(args.model, args.confidence)
    losses, forecasts, dates = roll_backtest(prices, model)

    # The 0/1 breach flags are a traced intermediate of count_breaches.
    summary = count_breaches(losses, forecasts)
    flags = np.array(summary.steps["is_breach"], dtype=float)

    fig, ax = plt.subplots(figsize=(11, 2.2))
    breach_timeline(flags, dates, ax=ax)
    save(fig, f"timeline_{args.model}.png")


if __name__ == "__main__":
    main()

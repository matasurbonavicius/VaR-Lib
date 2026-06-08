"""
Single chart: Basel traffic light -- the supervisory view of the breach count.

Green / yellow / red zones are derived from the binomial breach distribution; a
single line marks where this model's breach count falls.

    python examples/charts/traffic_light.py --model historical

Writes examples/output/traffic_light_<model>.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import load_prices, make_model, parse_args, roll_backtest, save  # noqa: E402
from varlib.backtest import basel_traffic_light, count_breaches  # noqa: E402
from varlib.plotting import traffic_light_chart  # noqa: E402
from varlib.plotting._style import get_pyplot  # noqa: E402


def main():
    args = parse_args(__doc__)
    plt = get_pyplot()

    prices = load_prices()
    model = make_model(args.model, args.confidence)
    losses, forecasts, _ = roll_backtest(prices, model)

    summary = count_breaches(losses, forecasts)
    light = basel_traffic_light(
        summary.n_breaches, summary.n_observations, args.confidence
    )

    fig, ax = plt.subplots(figsize=(8, 2.4))
    traffic_light_chart(result=light, ax=ax)
    save(fig, f"traffic_light_{args.model}.png")


if __name__ == "__main__":
    main()

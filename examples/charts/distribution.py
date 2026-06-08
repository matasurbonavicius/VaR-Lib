"""
Single chart: return distribution with the VaR and ES lines overlaid.

Shows where in the loss tail the two risk numbers sit, and why ES >= VaR. This
one is a point-in-time estimate on the most recent two years rather than a roll.

    python examples/charts/distribution.py --model historical

Writes examples/output/distribution_<model>.png
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from _common import load_prices, make_model, parse_args, save  # noqa: E402
from varlib._returns import to_returns  # noqa: E402
from varlib.plotting import distribution_chart  # noqa: E402
from varlib.plotting._style import get_pyplot  # noqa: E402


def main():
    args = parse_args(__doc__)
    plt = get_pyplot()

    prices = load_prices()
    # Estimate on the most recent ~two years of returns.
    recent = to_returns(prices.to_numpy())[-500:]
    result = make_model(args.model, args.confidence).run(returns=recent)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    distribution_chart(
        recent, result.value, result.expected_shortfall, args.confidence, ax=ax
    )
    save(fig, f"distribution_{args.model}.png")


if __name__ == "__main__":
    main()

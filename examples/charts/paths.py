"""
Single chart: the simulated paths behind a Monte-Carlo VaR.

Two stacked rows, one model each. Each draws 10,000 cumulative-return paths over
the holding period, with the distribution of where they end up attached on the
right -- the same h-day loss tail the model reads its VaR from.

    python examples/charts/paths.py

Writes examples/output/paths.png. The two simulations regenerate each model's
path data the way the model does internally (same fitted parameters / same
resampling), so the terminal tail matches the VaR the library would report.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from varlib.base import var_es_from_returns
from varlib.models.parametric.jump import estimate_jump_parameters, _sum_jumps
from varlib.plotting import paths_chart

HERE = os.path.dirname(os.path.dirname(__file__))
DATA = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT = os.path.join(HERE, "output")

CONFIDENCE, HORIZON, N_PATHS, SEED = 0.99, 21, 10_000, 0


def jump_paths(returns, rng):
    """(N_PATHS, HORIZON) daily returns from the fitted Merton jump-diffusion."""
    p = estimate_jump_parameters(returns)
    n = N_PATHS * HORIZON
    daily = rng.normal(p.mu_diffusion, p.sigma_diffusion, size=n)
    daily += _sum_jumps(rng.poisson(p.lambda_jump, size=n), p.mu_jump, p.sigma_jump, rng)
    return daily.reshape(N_PATHS, HORIZON)


def bootstrap_paths(returns, rng):
    """(N_PATHS, HORIZON) daily returns by resampling real history, like the bootstrap."""
    return rng.choice(returns, size=(N_PATHS, HORIZON), replace=True)


def main():
    prices = pd.read_csv(DATA, parse_dates=["Date"], index_col="Date")["AAPL"].dropna()
    returns = np.diff(np.log(prices.to_numpy()))[-500:]   # most recent ~two years
    rng = np.random.default_rng(SEED)
    os.makedirs(OUTPUT, exist_ok=True)

    rows = [
        ("Parametric jump-diffusion", jump_paths(returns, rng), "#1f77b4"),
        ("Historical bootstrap", bootstrap_paths(returns, rng), "#2ca02c"),
    ]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 8))
    for ax, (name, daily, color) in zip(axes, rows):
        var, _ = var_es_from_returns(daily.sum(axis=1), CONFIDENCE)  # VaR off the h-day sums
        paths_chart(daily, var, CONFIDENCE, title=name, color=color, ax=ax)

    fig.suptitle(f"{N_PATHS:,} simulated {HORIZON}-day paths -> the loss tail the VaR "
                 f"reads from (AAPL, {CONFIDENCE:.0%})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(OUTPUT, "paths.png"), dpi=120, bbox_inches="tight")
    print("wrote", os.path.join(OUTPUT, "paths.png"))


if __name__ == "__main__":
    main()

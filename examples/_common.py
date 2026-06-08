"""
Shared helpers for the example scripts.

Keeps the per-chart examples short and focused on the chart itself: loading the
sample price series, choosing a VaR model by name, and rolling a backtest live
once for the whole example set to reuse.
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from varlib import (
    HistoricalVar,
    HistoricalBootstrapVar,
    ParametricBrownianVar,
    ParametricOuVar,
    ParametricJumpVar,
    EwmaVar,
    rolling_backtest,
)

# The committed sample data lives next to this file, regardless of the cwd.
HERE = os.path.dirname(__file__)
DATA_PATH = os.path.join(HERE, "data", "AAPL.csv")
OUTPUT_DIR = os.path.join(HERE, "output")

# The models, keyed by the short name used on the command line. Each factory
# takes the confidence level and the holding-period horizon (in trading days), so
# the example can build any model at any horizon.
MODELS = {
    "historical": lambda c, h: HistoricalVar(c, horizon=h),
    "bootstrap": lambda c, h: HistoricalBootstrapVar(c, horizon=h, n_resamples=500),
    "brownian": lambda c, h: ParametricBrownianVar(c, horizon=h),
    "ou": lambda c, h: ParametricOuVar(c, horizon=h),
    "jump": lambda c, h: ParametricJumpVar(c, horizon=h, n_simulations=20_000),
    "ewma": lambda c, h: EwmaVar(c, horizon=h),
}

MODEL_LABELS = {
    "historical": "Historical",
    "bootstrap": "Historical bootstrap",
    "brownian": "Parametric Brownian",
    "ou": "Parametric OU",
    "jump": "Parametric jump",
    "ewma": "EWMA / RiskMetrics",
}


def load_prices() -> pd.Series:
    """Load the committed AAPL daily-close series, indexed by date."""
    df = pd.read_csv(DATA_PATH, parse_dates=["Date"], index_col="Date")
    return df["AAPL"].dropna()


def parse_args(description: str, extra_args=None) -> argparse.Namespace:
    """Standard CLI for every example: pick a model and a confidence level.

    `extra_args` is an optional callback taking the ArgumentParser, so a script
    can register its own flags (e.g. the report's --format) on top of the shared
    --model / --confidence pair.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="historical",
        help="VaR model to use (default: historical).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.99,
        help="Confidence level in (0, 1), e.g. 0.99 (default: 0.99).",
    )
    if extra_args is not None:
        extra_args(parser)
    return parser.parse_args()


def make_model(name: str, confidence: float, horizon: int = 1):
    """Build a model instance from its short name, at the given horizon."""
    return MODELS[name](confidence, horizon)


def roll_backtest(prices: pd.Series, model, window: int = 250, overlap: bool = True):
    """
    Roll a VaR model through the price history, at the model's own horizon.

    Thin wrapper over ``varlib.backtest.rolling_backtest`` -- the rolling helper
    now lives in the library itself, so the examples just hand it the AAPL price
    Series. Passing the Series means the end-of-period date labels come straight
    from its index. Returns aligned arrays of (losses, forecasts, dates); each
    date is the END of that step's h-day holding period.
    """
    return rolling_backtest(model, prices=prices, window=window, overlap=overlap)


def ensure_output_dir() -> str:
    """Create examples/output/ on first use and return its path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def save(fig, filename: str) -> str:
    """Save a figure into examples/output/ and report the path.

    Uses a tight bounding box, which is right for single charts where we want
    the image cropped to the content.
    """
    path = os.path.join(ensure_output_dir(), filename)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Saved {path}")
    return path


def save_page(fig, filename: str) -> str:
    """Save a fixed-page figure (e.g. A4) into examples/output/.

    Unlike `save`, this does NOT crop to a tight bounding box -- the whole page,
    including the margins the layout reserved, is preserved. The format follows
    the file extension, so the same figure can be written as both .pdf and .png.
    """
    path = os.path.join(ensure_output_dir(), filename)
    fig.savefig(path, dpi=200)
    print(f"Saved {path}")
    return path

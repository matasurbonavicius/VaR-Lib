"""Shared fixtures for the plotting tests.

All plotting tests are skipped if matplotlib is not installed (it is an optional
extra). The Agg backend is forced so the tests run headless on any machine.
"""

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")


@pytest.fixture
def backtest_data():
    """A small synthetic rolled backtest: losses, forecasts, and breach flags."""
    rng = np.random.default_rng(0)
    forecasts = np.full(300, 0.03)
    losses = rng.normal(0.0, 0.012, 300)
    # Force a handful of clear breaches so charts have something to mark.
    losses[[50, 120, 121, 122, 250]] = 0.05
    return losses, forecasts


@pytest.fixture(autouse=True)
def close_figures():
    """Close any figures a test opened, so they do not accumulate."""
    import matplotlib.pyplot as plt

    yield
    plt.close("all")

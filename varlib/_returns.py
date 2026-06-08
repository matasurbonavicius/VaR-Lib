"""
Price-to-return conversion, written step by step.

This is shared by every model that is handed a price series instead of returns.
It uses log returns, which is the standard choice for VaR work because they are
additive across time: the return over h days is simply the sum of the h daily
returns, which is how the models build their h-day (horizon) loss distributions.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def to_returns(prices: Any, steps: Optional[dict[str, Any]] = None) -> np.ndarray:
    """
    Convert a price series to log returns.

    Parameters
    ----------
    prices
        A price series (numpy array, list, or pandas Series). Must be strictly
        positive, because log returns are undefined for non-positive prices.
    steps
        Optional trace dictionary. If given, every intermediate is recorded.

    Returns
    -------
    np.ndarray
        A 1-D array of log returns, one shorter than the input.
    """
    if steps is None:
        steps = {}

    # Step 1: coerce the input to a clean 1-D float array.
    price_array = _to_array(prices)
    steps["prices"] = price_array

    # Step 2: prices must be positive for log returns to be defined.
    if np.any(price_array <= 0):
        raise ValueError("All prices must be strictly positive for log returns.")
    if price_array.size < 2:
        raise ValueError("Need at least two prices to compute one return.")

    # Step 3: take the ratio of consecutive prices.
    price_ratios = price_array[1:] / price_array[:-1]
    steps["price_ratios"] = price_ratios

    # Step 4: the log of each ratio is the log return for that period.
    log_returns = np.log(price_ratios)
    steps["returns"] = log_returns

    return log_returns


def _to_array(data: Any) -> np.ndarray:
    """Coerce a price input to a 1-D float array, dropping NaNs."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_numpy()
    arr = np.asarray(data, dtype=float).ravel()
    return arr[~np.isnan(arr)]

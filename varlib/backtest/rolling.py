"""
Rolling VaR -- turn one return (or price) series into a series of VaR forecasts.

This is the headline workflow of a single-series VaR library: walk a model
through history, and at each step fit it on a trailing window and record the VaR
it forecasts for the period ahead. The output is a *series* of forecasts, one per
step, which is exactly what every backtest in this package consumes.

Two functions, layered:

* ``rolling_var``      -- series in, series of VaR forecasts out.
* ``rolling_backtest`` -- the same roll, but it also lines up each forecast with
                          the loss that was *actually realised* over the matching
                          holding period, returning the aligned
                          ``(realised_losses, var_forecasts, dates)``
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from varlib._returns import to_returns
from varlib.base import VarModel


def rolling_var(
    model: VarModel,
    returns: Optional[Any] = None,
    prices: Optional[Any] = None,
    window: int = 250,
    step: int = 1,
    field: str = "value",
) -> np.ndarray:
    """
    Roll a VaR model through a series and return one forecast per step.

    At each step ``t`` (starting once a full look-back window exists) the model is
    fitted on the trailing ``window`` returns and its forecast is recorded.

    Parameters
    ----------
    model
        Any ``VarModel`` instance (Historical, EWMA, ...). Its ``horizon``
        is respected
    returns, prices
        Provide exactly one. ``prices`` is converted to log returns first. Both
        accept a numpy array, a list, or a pandas Series.
    window
        Trailing look-back length, in number of returns, used to fit the model.
    step
        How many returns to advance between forecasts. ``1`` (default) gives a
        forecast every day; ``horizon`` gives non-overlapping windows.
    field
        Which result attribute to collect: ``"value"`` (VaR) or
        ``"expected_shortfall"`` (ES).

    Returns
    -------
    np.ndarray
        The forecast series, one value per step.
    """
    rets = _as_returns(returns, prices)
    if field not in ("value", "expected_shortfall"):
        raise ValueError("field must be 'value' or 'expected_shortfall'.")
    if window < 1:
        raise ValueError("window must be >= 1.")
    if step < 1:
        raise ValueError("step must be >= 1.")
    if rets.size <= window:
        raise ValueError(
            f"Need more than window={window} returns to roll, got {rets.size}."
        )

    horizon = int(getattr(model, "horizon", 1))

    forecasts = []
    # Stop so a full h-day realised window could exist past each forecast: the
    # last fit uses returns[t-window:t] and the period it forecasts is
    # returns[t:t+horizon], so t may run up to len(rets) - horizon.
    for t in range(window, rets.size - horizon + 1, step):
        result = model.run(returns=rets[t - window:t])
        forecasts.append(getattr(result, field))
    return np.asarray(forecasts, dtype=float)


def rolling_backtest(
    model: VarModel,
    returns: Optional[Any] = None,
    prices: Optional[Any] = None,
    window: int = 250,
    overlap: bool = True,
    dates: Optional[Sequence[Any]] = None,
):
    """
    Roll a model through history and align each forecast with the realised loss.

    For each step past the look-back window this records:
      * the forecast VaR -- already an h-day figure (``h = model.horizon``); and
      * the realised h-day loss over the next ``h`` periods. Log returns are
        additive, so the h-day loss is ``-(returns[t] + ... + returns[t+h-1])``.

    Parameters
    ----------
    model
        Any ``VarModel`` instance; its ``horizon`` drives the realised window.
    returns, prices
        Provide exactly one. If a pandas Series of ``prices`` (or ``returns``) is
        given, its index is used to label each step (overriding ``dates``).
    window
        Trailing look-back length, in number of returns.
    overlap
        ``True`` (default) advances one period at a time, so the h-day windows
        overlap -- uses all the data and is the common charting convention, but
        the overlapping observations are serially dependent (which biases the
        Dynamic Quantile independence test). ``False``
        advances ``horizon`` periods at a time, giving non-overlapping,
        independent h-day returns -- cleaner statistics, but smaller sample. 
        Innacurate for large horizons and high confidence (99%+) because sample will be tiny.
    dates
        Optional labels for each underlying return. If omitted and a pandas Series
        was passed, that Series' index is used; otherwise an integer index.

    Returns
    -------
    (realised_losses, var_forecasts, dates) : tuple of np.ndarray, np.ndarray, list
        Aligned arrays plus the end-of-period date label for each step (the END
        of that step's h-day holding period).
    """
    rets = _as_returns(returns, prices)
    horizon = int(getattr(model, "horizon", 1))
    step = 1 if overlap else horizon
    if window < 1:
        raise ValueError("window must be >= 1.")
    if rets.size <= window + horizon - 1:
        raise ValueError(
            f"Need more than window+horizon-1={window + horizon - 1} returns to "
            f"form one step, got {rets.size}."
        )

    # Resolve the end-of-period label for each step. A step that starts at return
    # index t consumes returns t .. t+horizon-1, so the period ENDS at the price
    # whose position is t+horizon. `labels` and `offset` encode where that lands:
    # `labels[t + horizon - 1 + offset]` is the end-of-period label, or an integer
    # axis is used when there is no labelled input.
    labels, offset = _resolve_labels(returns, prices, dates, rets.size)

    forecasts, realised, step_dates = [], [], []
    for t in range(window, rets.size - horizon + 1, step):
        forecasts.append(model.run(returns=rets[t - window:t]).value)
        realised.append(-rets[t:t + horizon].sum())     # h-day cumulative loss
        if labels is not None:
            step_dates.append(labels[t + horizon - 1 + offset])
        else:
            step_dates.append(t + horizon)
    return np.asarray(realised), np.asarray(forecasts), step_dates


# -- helpers ----------------------------------------------------------------


def _as_returns(returns: Optional[Any], prices: Optional[Any]) -> np.ndarray:
    """Validate the (returns XOR prices) input and return a 1-D returns array."""
    if (returns is None) == (prices is None):
        raise ValueError("Provide exactly one of `returns` or `prices`.")
    if prices is not None:
        return to_returns(prices)
    arr = np.asarray(_values(returns), dtype=float).ravel()
    return arr


def _resolve_labels(returns, prices, dates, n_returns):
    """Return ``(labels, offset)`` for end-of-period date labels, or ``(None, 0)``.

    The end-of-period label for a step starting at return index ``t`` is the
    label of the price at position ``t + horizon``. We return the full label list
    and the ``offset`` that maps "last return index of the period" to it:

      * **Price labels** (length ``n_returns + 1``): the price at position
        ``t + horizon`` is the natural end-of-period stamp, so ``offset = 1``
        (``labels[(t + horizon - 1) + 1]``). This matches the convention of
        labelling an h-day loss by the date it is realised.
      * **Return labels** (length ``n_returns``): no extra price row exists, so we
        stamp by the last return in the period, ``offset = 0``.
    """
    import pandas as pd

    source = prices if prices is not None else returns
    if isinstance(source, (pd.Series, pd.DataFrame)):
        idx = list(source.index)
        if prices is not None and len(idx) == n_returns + 1:
            return idx, 1
        if returns is not None and len(idx) == n_returns:
            return idx, 0
        return None, 0
    if dates is not None:
        dates = list(dates)
        if len(dates) == n_returns + 1:      # labels the prices
            return dates, 1
        if len(dates) == n_returns:          # labels the returns
            return dates, 0
        return None, 0
    return None, 0


def _values(data: Any) -> Any:
    """Return the raw values of a pandas object, else the input unchanged."""
    import pandas as pd

    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    return data

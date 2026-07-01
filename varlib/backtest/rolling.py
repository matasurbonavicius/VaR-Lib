"""
Rolling VaR -- turn one return series into a series of VaR forecasts.

This is the headline workflow of a single-series VaR library: walk a model
through history, and at each step fit it on a trailing window and record the VaR
it forecasts for the period ahead. The output is a *series* of forecasts, one per
step, which is exactly what every backtest in this package consumes.

Two functions, layered:

* ``rolling_var``      -- series in, series of VaR forecasts out. The pure
                          "produce a VaR for each day" helper.
* ``rolling_backtest`` -- the same roll, but it also lines up each forecast with
                          the loss that was *actually realised* over the matching
                          holding period, returning the aligned
                          ``(realised_losses, var_forecasts, dates)`` triple the
                          backtests (Kupiec / DQ / traffic light) expect.

Horizon alignment is the methodological heart of a multi-day backtest: the model
already computes VaR *directly* at its horizon (see ``varlib.base``), and the
realised loss here is the loss over the **same** number of days, so forecast and
outcome are always on one footing -- never a 1-day VaR compared to a 10-day loss.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from varlib.base import VarModel


def rolling_var(
    model: VarModel,
    returns: Any,
    window: int = 250,
    step: int = 1,
    field: str = "value",
) -> np.ndarray:
    """
    Roll a VaR model through a series and return one forecast per step.

    At each step ``t`` (starting once a full look-back window exists) the model is
    fitted on the trailing ``window`` returns and its forecast is recorded. The
    forecast is read from ``result.<field>`` -- ``"value"`` for the VaR (default)
    or ``"expected_shortfall"`` for the ES -- so the same roll produces either a
    VaR or an ES series.

    Parameters
    ----------
    model
        Any ``VarModel`` instance (Historical, EWMA, ...). Its ``horizon``
        is respected: the forecast is already an h-day figure.
    returns
        The log-return series: a numpy array, a list, or a pandas Series. If you
        have prices, compute the log returns yourself first.
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
    rets = _as_returns(returns)
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
        result = model.run(rets[t - window:t])
        forecasts.append(getattr(result, field))
    return np.asarray(forecasts, dtype=float)


def rolling_backtest(
    model: VarModel,
    returns: Any,
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

    The forecast and the realised loss are therefore on the **same** holding
    period -- the alignment a multi-day backtest depends on.

    Parameters
    ----------
    model
        Any ``VarModel`` instance; its ``horizon`` drives the realised window.
    returns
        The log-return series. If a pandas Series is given, its index is used to
        label each step (overriding ``dates``). If you have prices, compute the
        log returns yourself first.
    window
        Trailing look-back length, in number of returns.
    overlap
        ``True`` (default) advances one period at a time, so the h-day windows
        overlap -- uses all the data and is the common charting convention, but
        the overlapping observations are serially dependent (which biases the
        Dynamic Quantile independence test). ``False``
        advances ``horizon`` periods at a time, giving non-overlapping,
        independent h-day returns -- cleaner statistics, ~1/h as many points.
    dates
        Optional labels for each underlying return. If omitted and a pandas Series
        of returns was passed, that Series' index is used; otherwise an integer
        index.

    Returns
    -------
    (realised_losses, var_forecasts, dates) : tuple of np.ndarray, np.ndarray, list
        Aligned arrays plus the end-of-period date label for each step (the END
        of that step's h-day holding period).
    """
    rets = _as_returns(returns)
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
    # index t consumes returns t .. t+horizon-1, so the period ENDS at the last
    # of those, return index t+horizon-1. `labels[t + horizon - 1]` is the
    # end-of-period label, or an integer axis is used when there is no labelled
    # input.
    labels = _resolve_labels(returns, dates, rets.size)

    forecasts, realised, step_dates = [], [], []
    for t in range(window, rets.size - horizon + 1, step):
        forecasts.append(model.run(rets[t - window:t]).value)
        realised.append(-rets[t:t + horizon].sum())     # h-day cumulative loss
        if labels is not None:
            step_dates.append(labels[t + horizon - 1])
        else:
            step_dates.append(t + horizon)
    return np.asarray(realised), np.asarray(forecasts), step_dates


# -- helpers ----------------------------------------------------------------


def _as_returns(returns: Any) -> np.ndarray:
    """Validate the returns input and return a 1-D returns array."""
    if returns is None:
        raise ValueError("`returns` is required.")
    return np.asarray(_values(returns), dtype=float).ravel()


def _resolve_labels(returns, dates, n_returns):
    """Return the per-return label list for end-of-period stamping, or ``None``.

    Each label positionally corresponds to one return, so the label for the step
    ending at return index ``i`` is ``labels[i]``. A pandas Series of returns
    carries the authoritative labels via its index and wins over an explicit
    ``dates`` argument; a bare ``dates`` sequence is used otherwise. Anything that
    does not line up one-to-one with the returns yields ``None`` (integer axis).
    """
    import pandas as pd

    if isinstance(returns, (pd.Series, pd.DataFrame)):
        idx = list(returns.index)
        return idx if len(idx) == n_returns else None
    if dates is not None:
        dates = list(dates)
        return dates if len(dates) == n_returns else None
    return None


def _values(data: Any) -> Any:
    """Return the raw values of a pandas object, else the input unchanged."""
    import pandas as pd

    if isinstance(data, (pd.Series, pd.DataFrame)):
        return data.to_numpy()
    return data

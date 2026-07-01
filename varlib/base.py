"""
The contract that every VaR model in this library obeys.

`VarResult`  -- the standard output: one number, plus every intermediate that
               was used to compute it.
`VarModel`   -- the standard input/output abstract class. A model is given a
               return series and must produce a `VarResult`. Subclasses
               implement exactly one method, `_compute`, which does the actual
               maths and fills the trace dictionary.

Returns are the currency of this library: every model consumes log returns and
nothing else. If you start from prices, compute the log returns yourself first
(e.g. ``np.log(prices / prices.shift(1)).dropna()``) and pass them in.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class VarResult:
    """
    The standard output of every VaR model.

    Attributes
    ----------
    value
        The Value at Risk, expressed as a positive loss fraction. For example
        0.023 means "we expect to lose no more than 2.3% over the horizon, with
        the stated confidence". Multiply by position value to get money.
    expected_shortfall
        The Expected Shortfall (a.k.a. CVaR / Conditional VaR), expressed as a
        positive loss fraction. It is the average loss on the days that breach
        the VaR -- "if it goes wrong, how bad is it on average?". ES is always
        at least as large as VaR.
    confidence
        The confidence level used, e.g. 0.99.
    horizon
        The holding period in number of return observations (1 = one period).
        The VaR is computed *directly* at this horizon -- e.g. a 10-day VaR is
        the quantile of the 10-day loss distribution -- not a one-day number
        scaled up. See ``VarModel`` for why.
    method
        The name of the model that produced this result.
    steps
        Every intermediate value used in the calculation, keyed by name.
        This is the audit trail.
    """

    value: float
    expected_shortfall: float
    confidence: float
    horizon: int
    method: str
    steps: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic only
        var_pct = f"{self.value * 100:.3f}%"
        es_pct = f"{self.expected_shortfall * 100:.3f}%"
        return (
            f"VarResult(method={self.method!r}, var={var_pct}, es={es_pct}, "
            f"confidence={self.confidence}, horizon={self.horizon})"
        )

    def explain(self) -> str:
        """Return a line-by-line trace of the calculation."""
        lines = [
            f"Method     : {self.method}",
            f"Confidence : {self.confidence}",
            f"Horizon    : {self.horizon} period(s)",
            f"VaR        : {self.value * 100:.4f}% loss",
            f"ES         : {self.expected_shortfall * 100:.4f}% loss",
            "Steps:",
        ]
        for name, val in self.steps.items():
            lines.append(f"  - {name}: {_short(val)}")
        return "\n".join(lines)


class VarModel(ABC):
    """
    Abstract base class for every VaR model.

    A model is constructed with its parameters (confidence, horizon, ...) and is
    then `run` against data. The base class handles everything that is common to
    all models -- input validation and cleaning -- so each concrete model only
    has to express its own formula.

    Subclasses implement `_compute(returns, steps)`:
      * `returns` is a clean 1-D numpy array of per-period (one-day) returns.
      * `steps`   is a fresh dict; fill it with every intermediate you produce.
      * read `self.horizon` and return the (VaR, ES) pair *at that horizon*, as
        two positive loss fractions.
    """

    #: Name of the method, set by each subclass.
    method_name: str = "abstract"

    def __init__(self, confidence: float = 0.99, horizon: int = 1) -> None:
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if horizon < 1 or int(horizon) != horizon:
            raise ValueError(f"horizon must be a positive integer, got {horizon}")
        self.confidence = float(confidence)
        self.horizon = int(horizon)

    # -- public API ---------------------------------------------------------

    def run(self, returns: Any) -> VarResult:
        """
        Compute VaR from a return series.

        `returns` is a series of per-period log returns (a numpy array, list, or
        pandas Series). If you have prices, compute the log returns yourself
        first (e.g. ``np.log(prices / prices.shift(1)).dropna()``).
        """
        steps: dict[str, Any] = {}

        # Step 1: obtain a clean 1-D array of per-period returns.
        clean_returns = self._prepare_returns(returns, steps)

        # Step 2: run the concrete model's formula on those returns. Every model
        # returns both its VaR and its ES (the average loss beyond the VaR),
        # already at the requested horizon (the model reads self.horizon).
        var, es = self._compute(clean_returns, steps)

        return VarResult(
            value=float(var),
            expected_shortfall=float(es),
            confidence=self.confidence,
            horizon=self.horizon,
            method=self.method_name,
            steps=steps,
        )

    # -- to be implemented by each model ------------------------------------

    @abstractmethod
    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        """
        Compute the (VaR, ES) at ``self.horizon`` from a clean one-day return array.

        Fill `steps` with every intermediate value. Return two positive loss
        fractions: the VaR and the Expected Shortfall, both at the model's
        horizon. For ``self.horizon == 1`` this is the ordinary one-day figure.
        """
        raise NotImplementedError

    # -- shared machinery ---------------------------------------------------

    def _prepare_returns(
        self,
        returns: Any,
        steps: dict[str, Any],
    ) -> np.ndarray:
        """Validate the input and return a clean 1-D array of returns."""
        if returns is None:
            raise ValueError("`returns` is required.")

        clean = _as_clean_array(returns)
        steps["returns"] = clean

        if clean.size == 0:
            raise ValueError("No return observations after cleaning.")
        steps["n_observations"] = int(clean.size)
        return clean


# -- shared VaR/ES definition -----------------------------------------------


def var_es_from_returns(
    returns: np.ndarray, confidence: float
) -> tuple[float, float]:
    """The empirical VaR and ES of a return sample, as positive loss fractions.

    Every model reduces to this once it has a sample of (real or simulated)
    returns, so the definition lives here and nowhere else:

      * VaR is the loss at the confidence-level quantile (linear interpolation
        between order statistics, the standard empirical-quantile estimator).
      * ES  is the average of the losses at or beyond that VaR -- the expected
        loss given a breach. If the tail is empty (tiny samples), it falls back
        to the VaR itself.

    A VaR is about losses, so returns are sign-flipped first: a -0.02 return is
    a +0.02 loss.
    """
    losses = -np.asarray(returns, dtype=float)
    var = float(np.quantile(losses, confidence, method="linear"))
    tail_losses = losses[losses >= var]
    es = float(np.mean(tail_losses)) if tail_losses.size else var
    return var, es


# -- shared horizon helpers -------------------------------------------------


def cumulative_returns(
    returns: np.ndarray, horizon: int, overlapping: bool = True
) -> np.ndarray:
    """Sum `returns` into `horizon`-period cumulative returns.

    Log returns are additive, so the cumulative return over `horizon` periods is
    the sum of the per-period returns in the window. There are two ways to lay
    the windows out:

    ``overlapping=True`` (default)
        Every window of `horizon` consecutive returns, stepping one period at a
        time. This uses all the data and gives the most stable empirical h-day
        distribution -- the standard construction for historical h-day VaR. The
        cost is that neighbouring windows share days, so the samples are
        autocorrelated. Length: ``len(returns) - horizon + 1``.

    ``overlapping=False``
        Disjoint back-to-back blocks of `horizon` returns. The windows share no
        days, so the h-day returns are statistically independent, but there are
        ~h times fewer of them, so the tail estimate is noisier. Any leftover
        days at the end that don't fill a whole block are dropped. Length:
        ``len(returns) // horizon``.

    For ``horizon == 1`` the input is returned unchanged either way.
    """
    r = np.asarray(returns, dtype=float)
    if horizon <= 1:
        return r
    if r.size < horizon:
        raise ValueError(
            f"Need at least horizon={horizon} returns to form one cumulative "
            f"window, got {r.size}."
        )

    if not overlapping:
        # Drop the leftover tail, then sum each disjoint block of `horizon` rows.
        n_blocks = r.size // horizon
        return r[: n_blocks * horizon].reshape(n_blocks, horizon).sum(axis=1)

    # A cumulative-sum trick gives every window sum in O(n): csum[i+h] - csum[i].
    csum = np.concatenate(([0.0], np.cumsum(r)))
    return csum[horizon:] - csum[:-horizon]


# -- small private helpers --------------------------------------------------


def _as_clean_array(data: Any) -> np.ndarray:
    """Coerce input to a 1-D float array and drop NaNs."""
    if isinstance(data, (pd.Series, pd.DataFrame)):
        data = data.to_numpy()
    arr = np.asarray(data, dtype=float).ravel()
    return arr[~np.isnan(arr)]


def _short(val: Any) -> str:
    """Compactly describe an intermediate value for the explain() trace."""
    if isinstance(val, np.ndarray):
        head = np.array2string(val[:3], precision=5)
        return f"array(shape={val.shape}, first={head})"
    if isinstance(val, float):
        return f"{val:.6f}"
    return repr(val)

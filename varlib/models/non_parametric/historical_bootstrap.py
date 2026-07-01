"""
Historical Bootstrap VaR -- resample the past, then read the quantile off each resample.

Plain Historical VaR reads one quantile off the single historical window we happen
to have. That single number carries no sense of how much it would wobble if we had
observed a slightly different window. The bootstrap answers exactly that: it draws
many synthetic samples from history (with replacement) and computes a plain
Historical VaR/ES on each. The average across resamples is the estimate; the spread
across resamples is its standard error -- the whole point of bootstrapping rather
than reading a single quantile.

It is still fully non-parametric: every resampled return is a real historical
return; nothing is fitted. We only assume the future looks like a reshuffling of the
past.

< Historical Bootstrap VaR assumes the future looks like a reshuffling of the past. >

Reference: Efron & Tibshirani (1993), "An Introduction to the Bootstrap", ch. 6
(the standard treatment of the bootstrap standard error).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel, var_es_from_returns


class HistoricalBootstrapVar(VarModel):
    """
    Bootstrapped historical-simulation VaR and ES, with a standard error.

    The exact sequence to derive Historical Bootstrap VaR and ES is:
    1. Take returns.
    2. Draw `n_resamples` synthetic samples from history with replacement; for
       horizon h each h-day return is the sum of h daily draws (log returns are
       additive).
    3. Take each resample's plain Historical VaR and ES (see `var_es_from_returns`).
    4. Average across resamples for the estimate; the spread across resamples is
       the standard error.
    """

    method_name = "historical_bootstrap"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        n_resamples: int = 1000,
        seed: int = 0,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        if n_resamples < 1:
            raise ValueError(f"n_resamples must be >= 1, got {n_resamples}")
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        steps["horizon"] = self.horizon

        # Each resample is a fresh n-long sample of h-day returns, drawn from
        # history with replacement. Its plain Historical VaR/ES is the bootstrap's
        # view of the h-day risk from one synthetic window (see _bootstrap_var_es).
        steps["seed"] = self.seed
        steps["n_resamples"] = self.n_resamples
        resample_vars, resample_es = _bootstrap_var_es(
            returns, self.confidence, self.horizon, self.n_resamples, self.seed
        )
        steps["resample_vars"] = resample_vars
        steps["resample_es"] = resample_es

        # The reported VaR and ES are the averages over all resamples; the spread
        # across resamples is the estimation standard error -- the main reason to
        # bootstrap rather than read one quantile.
        var = float(np.mean(resample_vars))
        es = float(np.mean(resample_es))
        steps["var"] = var
        steps["es"] = es
        steps["var_std_error"] = _std_error(resample_vars)
        steps["es_std_error"] = _std_error(resample_es)

        return var, es


def _bootstrap_var_es(
    returns: np.ndarray,
    confidence: float,
    horizon: int,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """The per-resample plain Historical VaR and ES of `n_resamples` bootstrap samples.

    Each resample is an `n`-long sample of h-day returns: we draw `horizon` daily
    returns from history with replacement and sum them (log returns are additive)
    to build one h-day return, `n` times. Drawing an ``(n, horizon)`` block at once
    and summing the rows is the vectorised form of "sum horizon daily draws, n
    times". The plain Historical VaR/ES of that resample (`var_es_from_returns`) is
    the bootstrap's view of the h-day risk from one synthetic window.

    Returns two arrays of length `n_resamples`: the per-resample VaRs and ESs. The
    generator is seeded so the whole draw is reproducible.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    rng = np.random.default_rng(seed)

    resample_vars = np.empty(n_resamples, dtype=float)
    resample_es = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        if horizon == 1:
            draw = rng.choice(r, size=n, replace=True)
        else:
            draw = rng.choice(r, size=(n, horizon), replace=True).sum(axis=1)
        resample_vars[i], resample_es[i] = var_es_from_returns(draw, confidence)
    return resample_vars, resample_es


def _std_error(estimates: np.ndarray) -> float:
    """The bootstrap standard error: the sample std of the per-resample estimates.

    With a single resample there is no spread to measure, so it is 0.0.
    """
    return float(np.std(estimates, ddof=1)) if estimates.size > 1 else 0.0


# -- functional API ---------------------------------------------------------
# Thin wrappers for callers who want the numbers without constructing a model.


def historical_bootstrap_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    n_resamples: int = 1000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """Historical Bootstrap VaR and ES as positive loss fractions.

    A thin functional wrapper around ``HistoricalBootstrapVar``. If ``steps`` is
    passed it is filled with the same trace the model records (``resample_vars``,
    ``var_std_error``, ...).
    """
    model = HistoricalBootstrapVar(
        confidence=confidence,
        horizon=horizon,
        n_resamples=n_resamples,
        seed=seed,
    )
    result = model.run(returns)
    if steps is not None:
        steps.update(result.steps)
    return result.value, result.expected_shortfall


def historical_bootstrap_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    n_resamples: int = 1000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> float:
    """Convenience wrapper returning only the bootstrapped VaR."""
    var, _ = historical_bootstrap_var_es(
        returns, confidence, n_resamples, seed, steps, horizon
    )
    return var

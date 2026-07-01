"""
Historical Bootstrap VaR -- historical simulation with resampling.

Plain Historical VaR reads the quantile off the one sample of returns we happen
to have. The bootstrap instead draws many new samples *from* that history (with
replacement), computes the VaR in each one, and averages. This smooths the
estimate and, as a by-product, tells us how uncertain the VaR is.

It is still fully non-parametric: every resampled return is a real historical
return. We only assume that the future looks like a reshuffling of the past.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel
from varlib.models.non_parametric.historical.historical import historical_var, historical_es


class HistoricalBootstrapVar(VarModel):
    """Bootstrapped historical-simulation VaR."""

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
            raise ValueError("n_resamples must be >= 1")
        self.n_resamples = int(n_resamples)
        self.seed = int(seed)

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        return historical_bootstrap_var_es(
            returns,
            self.confidence,
            self.n_resamples,
            self.seed,
            steps,
            horizon=self.horizon,
        )


def historical_bootstrap_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    n_resamples: int = 1000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Compute Historical Bootstrap VaR and ES at `horizon`, as positive loss fractions.

    Each resample is a fresh `n`-long sample of h-day returns: we draw `horizon`
    daily returns from history with replacement and sum them (log returns are
    additive) to build one h-day return, repeated `n` times. The plain Historical
    VaR and ES of that resample are then the bootstrap's view of the h-day risk.
    The reported numbers are the means across resamples; the spread across
    resamples is recorded as the estimation uncertainty.

    For ``horizon == 1`` each "h-day return" is a single daily draw, so this
    reduces exactly to the ordinary one-day bootstrap.
    """
    if steps is None:
        steps = {}

    returns = np.asarray(returns, dtype=float)
    n = returns.size
    steps["sample_size"] = int(n)
    horizon = int(horizon)
    if horizon > 1:
        steps["horizon"] = horizon

    # Step 1: set up a reproducible random generator.
    rng = np.random.default_rng(seed)
    steps["seed"] = int(seed)
    steps["n_resamples"] = int(n_resamples)

    # Step 2: in each resample, build n h-day returns -- each the sum of `horizon`
    # daily returns drawn with replacement -- then take that resample's plain
    # Historical VaR and ES. Drawing (n, horizon) at once and summing the rows is
    # the vectorised form of "sum horizon daily draws, n times".
    resample_vars = np.empty(n_resamples, dtype=float)
    resample_es = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        if horizon == 1:
            draw = rng.choice(returns, size=n, replace=True)
        else:
            draw = rng.choice(returns, size=(n, horizon), replace=True).sum(axis=1)
        draw_var = historical_var(draw, confidence)         # no per-draw trace
        resample_vars[i] = draw_var
        resample_es[i] = historical_es(draw, confidence, draw_var)
    steps["resample_vars"] = resample_vars
    steps["resample_es"] = resample_es

    # Step 3: the VaR and ES estimates are the averages over all resamples.
    var = float(np.mean(resample_vars))
    es = float(np.mean(resample_es))
    steps["var"] = var
    steps["es"] = es

    # Step 4: the spread across resamples measures how uncertain the estimates
    # are. This is the main reason to bootstrap rather than read one quantile.
    var_std_error = float(np.std(resample_vars, ddof=1)) if n_resamples > 1 else 0.0
    es_std_error = float(np.std(resample_es, ddof=1)) if n_resamples > 1 else 0.0
    steps["var_std_error"] = var_std_error
    steps["es_std_error"] = es_std_error

    return var, es


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

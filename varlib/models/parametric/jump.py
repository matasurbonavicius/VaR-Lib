"""
Parametric Jump-Diffusion VaR -- Merton's model, for fat-tailed returns.

A pure Brownian model has thin tails: it badly underestimates the chance of a
crash. Merton's jump-diffusion model adds rare, large jumps on top of the normal
day-to-day diffusion. Most days only the diffusion is active; occasionally a
jump fires and produces a big move. This fattens the tails and gives a more
honest VaR.

The model:
    return = diffusion (Normal) + (Poisson number of jumps) x (Normal jump size)

We separate jumps from ordinary moves with a simple, transparent threshold rule:
returns beyond a few standard deviations are treated as jumps; the rest are the
diffusion. From the two groups we estimate all parameters, then read the VaR off
the resulting mixture distribution by simulation (using a fixed seed, so the
result is reproducible and inspectable).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from varlib.base import VarModel, var_es_from_returns


class JumpParameters(NamedTuple):
    """Estimated Merton jump-diffusion parameters (per period)."""

    mu_diffusion: float     # drift of the ordinary (non-jump) returns
    sigma_diffusion: float  # volatility of the ordinary returns
    lambda_jump: float      # expected number of jumps per period
    mu_jump: float          # mean jump size
    sigma_jump: float       # volatility of jump sizes


class ParametricJumpVar(VarModel):
    """Merton jump-diffusion parametric VaR."""

    method_name = "parametric_jump"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        jump_threshold: float = 3.0,
        n_simulations: int = 50_000,
        seed: int = 0,
        params: JumpParameters | None = None,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        self.jump_threshold = float(jump_threshold)
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)
        self.params_override = params

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        return parametric_jump_var_es(
            returns,
            self.confidence,
            self.jump_threshold,
            self.n_simulations,
            self.seed,
            self.params_override,
            steps,
            horizon=self.horizon,
        )


def parametric_jump_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    jump_threshold: float = 3.0,
    n_simulations: int = 50_000,
    seed: int = 0,
    params: JumpParameters | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> tuple[float, float]:
    """
    Compute Merton jump-diffusion VaR and ES at `horizon`, as positive loss fractions.

    The mixture has no simple closed-form quantile, so the tail is found by Monte
    Carlo. To get the h-day risk we simulate an h-day *path*: each simulated h-day
    return is the sum of `horizon` independent daily returns, every day drawn from
    the same fitted diffusion-plus-jumps mixture. Taking the quantile of those
    h-day sums captures how the fat tails accumulate (and partly average out) over
    the holding period -- no sqrt-of-time scaling. Both VaR and ES are read off
    the same simulated h-day sample, so they stay consistent. For ``horizon == 1``
    each path is a single day, recovering the one-day result.
    """
    if steps is None:
        steps = {}

    returns = np.asarray(returns, dtype=float)
    horizon = int(horizon)

    # Step 1: get the jump-diffusion parameters (estimate or override).
    if params is None:
        params = estimate_jump_parameters(returns, jump_threshold, steps)
        steps["params_source"] = "estimated"
    else:
        steps["params_source"] = "override"
    steps["lambda_jump"] = params.lambda_jump
    steps["mu_jump"] = params.mu_jump
    steps["sigma_jump"] = params.sigma_jump

    # Step 2: simulate many h-day returns from the fitted mixture. We draw
    # n_simulations * horizon daily returns and sum each block of `horizon` to
    # form one h-day path return.
    rng = np.random.default_rng(seed)
    steps["seed"] = int(seed)
    steps["n_simulations"] = int(n_simulations)
    if horizon > 1:
        steps["horizon"] = horizon

    n_days = n_simulations * horizon
    # Diffusion part: one Normal draw per simulated day.
    diffusion = rng.normal(params.mu_diffusion, params.sigma_diffusion, size=n_days)
    # Jump part: a Poisson count of jumps per day, each a Normal draw, summed.
    jump_counts = rng.poisson(params.lambda_jump, size=n_days)
    jump_totals = _sum_jumps(jump_counts, params.mu_jump, params.sigma_jump, rng)
    daily_returns = diffusion + jump_totals
    # Sum each consecutive block of `horizon` days into one h-day return.
    if horizon > 1:
        simulated_returns = daily_returns.reshape(n_simulations, horizon).sum(axis=1)
    else:
        simulated_returns = daily_returns
    steps["simulated_returns"] = simulated_returns

    # Steps 3-4: VaR is the empirical quantile of the simulated losses; ES is
    # the average simulated loss at or beyond that VaR.
    var, es = var_es_from_returns(simulated_returns, confidence)
    steps["var"] = var
    steps["es"] = es

    return var, es


def parametric_jump_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    jump_threshold: float = 3.0,
    n_simulations: int = 50_000,
    seed: int = 0,
    params: JumpParameters | None = None,
    steps: dict[str, Any] | None = None,
    horizon: int = 1,
) -> float:
    """Convenience wrapper returning only the jump-diffusion VaR."""
    var, _ = parametric_jump_var_es(
        returns, confidence, jump_threshold, n_simulations, seed, params, steps, horizon
    )
    return var


def estimate_jump_parameters(
    returns: np.ndarray,
    jump_threshold: float = 3.0,
    steps: dict[str, Any] | None = None,
) -> JumpParameters:
    """
    Estimate Merton parameters by splitting returns into jumps and diffusion.

    A return is labelled a jump if it lies more than `jump_threshold` robust
    standard deviations from the median. Everything else is diffusion.
    """
    if steps is None:
        steps = {}

    returns = np.asarray(returns, dtype=float)
    n = returns.size
    if n < 5:
        raise ValueError("Need at least 5 observations to separate jumps.")

    # Step 1: a robust centre and spread, so the threshold is not itself
    # inflated by the jumps we are trying to detect.
    centre = float(np.median(returns))
    mad = float(np.median(np.abs(returns - centre)))
    robust_sigma = 1.4826 * mad  # MAD-to-sigma constant for normal data
    steps["robust_centre"] = centre
    steps["robust_sigma"] = robust_sigma

    # Step 2: flag the jumps.
    distance = np.abs(returns - centre)
    is_jump = distance > jump_threshold * robust_sigma
    steps["jump_threshold"] = float(jump_threshold)
    steps["n_jumps"] = int(np.sum(is_jump))

    jumps = returns[is_jump]
    diffusion = returns[~is_jump]

    # Step 3: diffusion parameters from the non-jump returns.
    mu_diffusion = float(np.mean(diffusion))
    sigma_diffusion = float(np.std(diffusion, ddof=1)) if diffusion.size > 1 else 0.0
    steps["mu_diffusion"] = mu_diffusion
    steps["sigma_diffusion"] = sigma_diffusion

    # Step 4: jump frequency and jump-size distribution.
    lambda_jump = float(jumps.size) / float(n)  # expected jumps per period
    if jumps.size >= 2:
        mu_jump = float(np.mean(jumps))
        sigma_jump = float(np.std(jumps, ddof=1))
    elif jumps.size == 1:
        mu_jump = float(jumps[0])
        sigma_jump = robust_sigma  # one jump: borrow the diffusion-scale spread
    else:
        mu_jump = 0.0
        sigma_jump = 0.0
    steps["lambda_jump"] = lambda_jump
    steps["mu_jump"] = mu_jump
    steps["sigma_jump"] = sigma_jump

    return JumpParameters(
        mu_diffusion=mu_diffusion,
        sigma_diffusion=sigma_diffusion,
        lambda_jump=lambda_jump,
        mu_jump=mu_jump,
        sigma_jump=sigma_jump,
    )


def _sum_jumps(
    counts: np.ndarray,
    mu_jump: float,
    sigma_jump: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    For each simulated period, sum `count` jump draws.

    A Normal jump size summed `k` independent times is itself Normal with mean
    k*mu and variance k*sigma^2, so we draw the total directly. This is exact
    and avoids a Python loop over individual jumps.
    """
    totals = np.zeros(counts.shape, dtype=float)
    has_jump = counts > 0
    k = counts[has_jump].astype(float)
    means = k * mu_jump
    stds = np.sqrt(k) * sigma_jump
    totals[has_jump] = rng.normal(means, stds)
    return totals

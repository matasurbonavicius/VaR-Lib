"""
Age-weighted Historical VaR -- historical simulation, but recent days count more.

Plain Historical VaR gives every day in the window the same weight: a crash 250
days ago pulls the quantile exactly as hard as a crash yesterday, and the number
only moves when an extreme day enters or leaves the window (the "ghosting" /
plateau effect). Age-weighting promises to fix both at once by weighting the latest 
returns more heavily than older ones, so VaR moves more smoothly and reacts faster to recent market conditions.

The VaR is then read off this *weighted* empirical distribution instead of the
flat one. Everything else is identical to Historical VaR -- it is still fully
non-parametric (every scenario is a real historical return; nothing is fitted).
The single knob `lambda` (the decay, in (0, 1]) is *chosen*, not estimated:

    lambda = 1        -> every weight equal        -> exactly plain Historical VaR
    lambda -> 0       -> only the newest days matter (very short memory)
    lambda ~ 0.97-0.99 -> the usual practical range

< Age-weighted VaR still assumes the future looks like the past, but it assumes
  the *recent* past looks more like it than the distant past. >

Reference: Boudoukh, Richardson & Whitelaw (1998), "The Best of Both Worlds":
The_Best_of_Both_Worlds_aw_historical.pdf
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel, cumulative_returns


class AgeWeightedHistoricalVar(VarModel):
    """
    Age-weighted (BRW) historical-simulation VaR and ES.

    This is the BRW three-step recipe (Boudoukh, Richardson & Whitelaw 1998,
    pp. 6-7), with an h-day extension:
    1. Take returns.
    2. Sum h-day returns -> for horizon 30, a rolling 30-day window sum.
    3. Step 1 of BRW: assign each return a geometric age weight `lambda^age`,
       newest gets the most, normalised to sum to 1 (the paper's `(1-lambda)/
       (1-lambda^K)` constant is just this normaliser).
    4. Steps 2-3 of BRW: order the losses, then read the VaR off the *weighted*
       distribution with half-weight centering and linear interpolation between
       adjacent points (see `_weighted_var_es`). The weighted
       average loss beyond the VaR is the ES.

    With `lambda_decay = 1` the weights are flat and this reduces to plain
    `HistoricalVar` (up to the quantile convention).

    test.historical.test_age_weighted_with_lambda_one_matches_historical tests if 
    historical simulation = age-weighted historical simulation with lambda=1.0.
    """

    method_name = "age_weighted_historical"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        lambda_decay: float = 0.98,
        overlapping: bool = True,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        if not 0.0 < lambda_decay <= 1.0:
            raise ValueError(
                f"lambda_decay must be in (0, 1], got {lambda_decay}"
            )
        self.lambda_decay = float(lambda_decay)
        self.overlapping = bool(overlapping)

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        # As in plain Historical VaR, the h-day VaR is the quantile of the h-day
        # loss distribution, read off h-day cumulative returns. For horizon 1
        # this is just the returns themselves.
        horizon_returns = cumulative_returns(
            returns, self.horizon, self.overlapping
        )

        steps["horizon"] = self.horizon
        if self.horizon > 1:
            steps["overlapping"] = self.overlapping
            steps["horizon_returns"] = horizon_returns

        # The only difference from Historical VaR: geometric age weights rather
        # than a flat 1/n. `weights` is oldest-first so it lines up with the
        # chronological `horizon_returns` (index 0 is the oldest observation).
        weights = _age_weights(horizon_returns.size, self.lambda_decay)
        steps["lambda_decay"] = self.lambda_decay
        steps["weights"] = weights

        # VaR is the loss at the confidence level of the *weighted* empirical
        # distribution; ES is the weighted average loss at or beyond it (see
        # _weighted_var_es for the definition).
        steps["losses"] = -np.asarray(horizon_returns, dtype=float)
        var, es = _weighted_var_es(horizon_returns, weights, self.confidence)
        steps["var"] = var
        steps["es"] = es

        return var, es


def _age_weights(n: int, lambda_decay: float) -> np.ndarray:
    """Normalised geometric age weights, oldest first.

    The newest observation (last index) gets the largest weight; each step back
    in time multiplies the weight by `lambda_decay`. The raw weights are
    `lambda^(age)` and are normalised to sum to 1. For `lambda_decay == 1` this
    is a flat `1/n`, which recovers plain Historical VaR exactly.
    """
    # age = 0 for the newest observation, n-1 for the oldest. Build newest-first
    # then reverse so index 0 is the oldest, matching the return series order.
    raw_newest_first = lambda_decay ** np.arange(n, dtype=float)
    raw = raw_newest_first[::-1]
    return raw / raw.sum()


def _weighted_var_es(
    returns: np.ndarray, weights: np.ndarray, confidence: float
) -> tuple[float, float]:
    """The empirical VaR and ES of a *weighted* return sample.

    The weighted counterpart of ``base.var_es_from_returns``: instead of every
    observation carrying the same 1/n probability, each return carries its own
    probability ``weights[i]`` (positive, summing to 1). The VaR is read off the
    *weighted* empirical distribution exactly as in the three-step recipe of
    Boudoukh, Richardson & Whitelaw (1998), "The Best of Both Worlds", pp. 6-7:

      * Step 2 -- order the returns in ascending order (i.e. losses descending).
      * Step 3 -- "start from the lowest return and keep accumulating the weights
        until x% is reached. Linear interpolation is used between adjacent points
        to achieve exactly x% of the distribution."

    ES is the weight-average of the losses at or beyond the VaR -- the expected
    loss given a breach, weighting recent breaches more. The paper does not
    define a weighted ES; this is the natural age-weighted analogue.

    As in ``base.var_es_from_returns``, returns are sign-flipped first: a -0.02
    return is a +0.02 loss.
    """
    losses = -np.asarray(returns, dtype=float)
    w = np.asarray(weights, dtype=float)

    # Step 2: order losses from largest (worst) to smallest, carrying each loss's weight with it. 
    order = np.argsort(losses)[::-1]
    sorted_losses = losses[order]
    sorted_w = w[order]

    # Half-weight centering: the running total *before* each loss, plus half of
    # the loss's own weight, is the cumulative probability the paper assigns to
    # that observation (p. 6). cum_before[i] = sum of weights of all worse losses.
    cum_before = np.cumsum(sorted_w) - sorted_w
    centered = cum_before + 0.5 * sorted_w

    # Step 3: walk down from the worst loss and linearly interpolate the loss
    # whose centered cumulative weight equals the tail probability (1 - conf).
    var = _interp_at_tail(sorted_losses, centered, 1.0 - confidence)

    tail_mask = losses >= var
    tail_weight = w[tail_mask].sum()
    es = (
        float(np.dot(losses[tail_mask], w[tail_mask]) / tail_weight)
        if tail_weight > 0
        else var
    )
    return var, es


def _interp_at_tail(
    sorted_losses: np.ndarray, centered: np.ndarray, tail_prob: float
) -> float:
    """Linearly interpolate the loss at cumulative weight ``tail_prob``.

    ``sorted_losses`` runs worst-first and ``centered`` is their (increasing)
    centered cumulative weight. Implements the interpolation on p. 7 of BRW:
    find the pair of observations whose centered weights straddle ``tail_prob``
    and interpolate the loss linearly between them. Outside the range of the
    centered weights (the extreme corners of the tail) we clamp to the nearest
    observed loss, since there is no adjacent point to interpolate against.
    """
    # centered is strictly increasing worst->best; np.interp needs it ascending.
    if tail_prob <= centered[0]:
        return float(sorted_losses[0])
    if tail_prob >= centered[-1]:
        return float(sorted_losses[-1])
    return float(np.interp(tail_prob, centered, sorted_losses))

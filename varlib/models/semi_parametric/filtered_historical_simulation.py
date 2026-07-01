"""
Filtered Historical Simulation (FHS) VaR -- historical simulation, de-volatilised.

Plain Historical VaR resamples raw past returns, which quietly assumes every day
was drawn from the same distribution. Markets don't work like that: volatility
comes in clusters (a calm 2017 day and a March-2020 day are not exchangeable), so
a raw resample mixes quiet-regime and crisis-regime days into one pot and reacts
to today's regime only slowly, as extreme days enter and leave the window.

FHS fixes exactly that by splitting each return into "how volatile was that day"
and "how bad was it *for* that volatility":

    1. Filter -- fit a volatility model (GARCH) and divide each return by its own
       conditional volatility. What remains, the standardized residual
       z_t = (r_t - mu) / sigma_t, is (close to) i.i.d.: the volatility clustering
       has been filtered out, so *these* are exchangeable and safe to resample.
    2. Simulate -- forecast the volatility forward over the horizon, draw
       standardized residuals from history with replacement, and re-inflate each
       by the *forecast* volatility. This grafts the shape of past shocks onto
       today's volatility level.
    3. Read the VaR/ES off the simulated horizon returns, exactly as every other
       model does.

The result is still non-parametric in its tail -- every shock is a real
standardized historical residual, nothing about the *shape* of the tail is
assumed -- but it is filtered through a parametric volatility model. That mix is
why FHS is called semi-parametric: parametric dynamics, empirical shocks.

< FHS assumes tomorrow's *shock* looks like a past shock, but scaled to today's
  volatility rather than the volatility of the day the shock happened on. >

Reference: Barone-Adesi, Giannopoulos & Vosper (1999), "VaR without correlations
for portfolios of derivative securities", Journal of Futures Markets 19(5),
583-602 -- the paper that introduced FHS. The GARCH-filter-then-bootstrap recipe
followed here is the one worked through in the MathWorks example "Using
Bootstrapping and Filtered Historical Simulation to Evaluate Market Risk".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from varlib.base import VarModel, var_es_from_returns

# arch supplies the GARCH filter. It is an optional dependency: importing this
# module without it raises a clear message pointing at the extra, rather than a
# bare ModuleNotFoundError from deep inside `_compute`.
try:
    from arch import arch_model
except ModuleNotFoundError as exc:  # pragma: no cover - exercised via message only
    arch_model = None
    _ARCH_IMPORT_ERROR = exc
else:
    _ARCH_IMPORT_ERROR = None

# arch's optimiser is conditioned for percent-scale returns (e.g. 1.3, not 0.013);
# on raw fractions the likelihood is so flat that it warns and underfits. We scale
# returns up by this factor for the fit and divide the same factor back out of
# every volatility, so nothing leaves this module in percent units.
_PERCENT = 100.0


class FilteredHistoricalSimulationVar(VarModel):
    """
    Filtered Historical Simulation (BGV) VaR and ES.

    The exact sequence to derive FHS VaR and ES is:
    1. Take returns.
    2. Fit a GARCH model and filter: divide each return by its conditional
       volatility to get the standardized residuals z_t (the de-volatilised
       shocks), which are close to i.i.d. and therefore safe to resample.
    3. Forecast the conditional volatility forward over the horizon.
    4. Simulate horizon returns: draw standardized residuals with replacement,
       re-inflate each by the forecast volatility, and sum over the horizon.
    5. VaR = the loss quantile of the simulated sample; ES = the average loss
       beyond it (the Historical definition, same as every other model).

    The volatility model is configurable through the `arch` library: `vol`
    ("Garch", "EGARCH", "GJR" via ``vol="Garch", o=1``), the orders `p`/`o`/`q`,
    and the innovation `dist`. The default is GARCH(1,1) with normal innovations
    -- the standard FHS filter -- but the residuals are bootstrapped either way,
    so `dist` only shapes the filter, never the simulated tail.
    """

    method_name = "filtered_historical_simulation"

    def __init__(
        self,
        confidence: float = 0.99,
        horizon: int = 1,
        vol: str = "Garch",
        p: int = 1,
        o: int = 0,
        q: int = 1,
        dist: str = "normal",
        mean: str = "Constant",
        n_simulations: int = 10_000,
        seed: int = 0,
    ) -> None:
        super().__init__(confidence=confidence, horizon=horizon)
        if arch_model is None:
            raise ImportError(
                "FilteredHistoricalSimulationVar needs the `arch` package for the "
                "GARCH filter. Install it with `pip install arch` (or "
                "`pip install varlib[fhs]`)."
            ) from _ARCH_IMPORT_ERROR
        if n_simulations < 1:
            raise ValueError(f"n_simulations must be >= 1, got {n_simulations}")
        self.vol = str(vol)
        self.p = int(p)
        self.o = int(o)
        self.q = int(q)
        self.dist = str(dist)
        self.mean = str(mean)
        self.n_simulations = int(n_simulations)
        self.seed = int(seed)

    def _compute(
        self, returns: np.ndarray, steps: dict[str, Any]
    ) -> tuple[float, float]:
        if self.horizon > 1:
            steps["horizon"] = self.horizon

        # Step 1: filter. Fit the GARCH model and split each return into its
        # conditional volatility and its standardized residual (see _garch_filter).
        # The residuals are the de-volatilised shocks we are allowed to resample.
        mu, residuals, forecast_sigma = _garch_filter(
            returns,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            dist=self.dist,
            mean=self.mean,
            horizon=self.horizon,
        )
        steps["mu"] = mu
        steps["standardized_residuals"] = residuals
        steps["forecast_sigma"] = forecast_sigma

        # Step 2: simulate horizon returns. Each simulated path draws `horizon`
        # standardized residuals with replacement and re-inflates them by the
        # forecast volatility for each step, then sums (see _simulate_returns).
        rng = np.random.default_rng(self.seed)
        steps["seed"] = self.seed
        steps["n_simulations"] = self.n_simulations
        simulated_returns = _simulate_returns(
            mu, residuals, forecast_sigma, self.n_simulations, rng
        )
        steps["simulated_returns"] = simulated_returns

        # Step 3: VaR = loss quantile of the simulated sample; ES = average loss
        # beyond it. Identical definition to every other model.
        var, es = var_es_from_returns(simulated_returns, self.confidence)
        steps["var"] = var
        steps["es"] = es

        return var, es


def _garch_filter(
    returns: np.ndarray,
    vol: str = "Garch",
    p: int = 1,
    o: int = 0,
    q: int = 1,
    dist: str = "normal",
    mean: str = "Constant",
    horizon: int = 1,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit a GARCH model and return the filter's three outputs.

    The filter separates "how volatile was the market" from "how big was the
    shock". Concretely it returns:

      * ``mu`` -- the fitted mean return (per period), added back to every
        simulated day so the drift is preserved.
      * ``residuals`` -- the standardized residuals ``z_t = (r_t - mu) / sigma_t``,
        an approximately i.i.d. sample of unit-scale shocks. These are what the
        bootstrap resamples: the volatility clustering has been divided out, so
        unlike the raw returns they are exchangeable.
      * ``forecast_sigma`` -- the conditional volatility forecast for each of the
        next ``horizon`` periods, ``(sigma_{T+1}, ..., sigma_{T+h})``. Simulated
        shocks are re-inflated by these, which is what makes FHS react to *today's*
        volatility rather than the window average.

    Everything is returned in the caller's return units: `arch` is fitted on
    percent-scaled returns for numerical stability (see ``_PERCENT``) and the
    scale is divided back out here.
    """
    r = np.asarray(returns, dtype=float)

    # Fit on percent-scaled returns; arch's optimiser is conditioned for that scale.
    model = arch_model(
        r * _PERCENT, mean=mean, vol=vol, p=p, o=o, q=q, dist=dist
    )
    fit = model.fit(disp="off")

    # mu and the standardized residuals are scale-free in the two senses we need:
    # z_t = (r_t - mu)/sigma_t is a pure number (the _PERCENT cancels), and mu is
    # divided back to return units. std_resid can carry NaNs for the first few
    # observations while the recursion warms up; drop them before resampling.
    mu = float(fit.params["mu"]) / _PERCENT
    residuals = np.asarray(fit.std_resid, dtype=float)
    residuals = residuals[~np.isnan(residuals)]

    # Conditional-volatility forecast for the next `horizon` periods. arch reports
    # variances in percent^2, so sqrt then divide by _PERCENT to reach return units.
    forecast = fit.forecast(horizon=horizon, reindex=False)
    forecast_variance = np.asarray(forecast.variance.values[-1], dtype=float)
    forecast_sigma = np.sqrt(forecast_variance) / _PERCENT

    return mu, residuals, forecast_sigma


def _simulate_returns(
    mu: float,
    residuals: np.ndarray,
    forecast_sigma: np.ndarray,
    n_simulations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate `n_simulations` horizon returns by the FHS bootstrap.

    For a horizon of ``h`` each simulated path is

        R = sum_{k=1..h} ( mu + sigma_{T+k} * z_k ),

    where every ``z_k`` is a standardized residual drawn from history with
    replacement and ``sigma_{T+k}`` is the forecast volatility for step ``k``.
    Re-inflating the *same* pool of empirical shocks by the *forecast* volatility
    is the whole idea of FHS: the shape of the tail is historical, its scale is
    today's. Drawing an ``(n, h)`` block of residuals at once and re-inflating
    column ``k`` by ``sigma_{T+k}`` is the vectorised form of that sum.
    """
    horizon = forecast_sigma.size
    # (n_simulations, horizon) standardized shocks, resampled with replacement.
    draws = rng.choice(residuals, size=(n_simulations, horizon), replace=True)
    # Re-inflate each column by its own step's forecast volatility, add the drift,
    # then sum across the horizon to form one horizon return per row.
    daily = mu + draws * forecast_sigma[np.newaxis, :]
    return daily.sum(axis=1)


def filtered_historical_simulation_var_es(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    vol: str = "Garch",
    p: int = 1,
    o: int = 0,
    q: int = 1,
    dist: str = "normal",
    mean: str = "Constant",
    n_simulations: int = 10_000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """Filtered Historical Simulation VaR and ES as positive loss fractions.

    A thin functional wrapper around ``FilteredHistoricalSimulationVar``. If
    ``steps`` is passed it is filled with the same trace the model records
    (``standardized_residuals``, ``forecast_sigma``, ...).
    """
    model = FilteredHistoricalSimulationVar(
        confidence=confidence,
        horizon=horizon,
        vol=vol,
        p=p,
        o=o,
        q=q,
        dist=dist,
        mean=mean,
        n_simulations=n_simulations,
        seed=seed,
    )
    result = model.run(returns)
    if steps is not None:
        steps.update(result.steps)
    return result.value, result.expected_shortfall


def filtered_historical_simulation_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    n_simulations: int = 10_000,
    seed: int = 0,
    steps: dict[str, Any] | None = None,
) -> float:
    """Convenience wrapper returning only the FHS VaR."""
    var, _ = filtered_historical_simulation_var_es(
        returns,
        confidence=confidence,
        horizon=horizon,
        n_simulations=n_simulations,
        seed=seed,
        steps=steps,
    )
    return var

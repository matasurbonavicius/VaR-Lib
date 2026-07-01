"""Tests for Filtered Historical Simulation (FHS) VaR.

FHS needs the optional `arch` package; the whole module is skipped if it is not
installed. arch's optimiser is noisy on synthetic data, so its convergence
warnings are filtered here -- they are not what these tests are about.
"""

import warnings

import numpy as np
import pytest

arch = pytest.importorskip("arch")

from varlib import (
    FilteredHistoricalSimulationVar,
    HistoricalVar,
    filtered_historical_simulation_var,
    filtered_historical_simulation_var_es,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _garch_returns(n=2000, seed=0):
    """A fat-tailed, volatility-clustering return series to filter."""
    rng = np.random.default_rng(seed)
    # A crude GARCH-like series: volatility that drifts and clusters.
    sigma = np.empty(n)
    sigma[0] = 0.01
    shocks = rng.standard_normal(n)
    r = np.empty(n)
    for t in range(n):
        if t > 0:
            sigma[t] = np.sqrt(1e-6 + 0.9 * sigma[t - 1] ** 2 + 0.05 * r[t - 1] ** 2)
        r[t] = sigma[t] * shocks[t]
    return r


def test_model_wrapper_runs():
    result = FilteredHistoricalSimulationVar(0.99, n_simulations=2000, seed=0).run(
        returns=_garch_returns()
    )
    assert result.method == "filtered_historical_simulation"
    assert result.value > 0


def test_es_at_least_var():
    result = FilteredHistoricalSimulationVar(0.99, n_simulations=5000, seed=0).run(
        returns=_garch_returns()
    )
    assert result.expected_shortfall >= result.value


def test_is_reproducible_with_seed():
    returns = _garch_returns()
    a = filtered_historical_simulation_var(returns, 0.99, seed=42, n_simulations=3000)
    b = filtered_historical_simulation_var(returns, 0.99, seed=42, n_simulations=3000)
    assert a == b


def test_functional_wrapper_fills_steps():
    steps = {}
    var, es = filtered_historical_simulation_var_es(
        _garch_returns(), 0.99, n_simulations=3000, seed=0, steps=steps
    )
    assert var > 0 and es >= var
    # The FHS trace exposes the two quantities that make it FHS.
    assert steps["standardized_residuals"].ndim == 1
    assert steps["forecast_sigma"].size == 1  # horizon 1


def test_standardized_residuals_are_roughly_unit_scale():
    """The filter should divide out the volatility, leaving ~unit-variance shocks."""
    result = FilteredHistoricalSimulationVar(0.99, n_simulations=1000, seed=0).run(
        returns=_garch_returns()
    )
    z = result.steps["standardized_residuals"]
    # Not exactly 1 on finite, mis-specified data, but far from the raw return scale.
    assert 0.5 < float(np.std(z)) < 1.5


def test_horizon_scales_up_the_var():
    returns = _garch_returns()
    var1 = FilteredHistoricalSimulationVar(0.99, horizon=1, seed=0).run(
        returns=returns
    ).value
    result10 = FilteredHistoricalSimulationVar(0.99, horizon=10, seed=0).run(
        returns=returns
    )
    assert result10.steps["forecast_sigma"].size == 10
    assert result10.value > var1


def test_reacts_to_a_recent_volatility_spike():
    """FHS re-inflates by the *forecast* vol, so a calm tail then a violent tail
    should give a higher VaR than the calm tail alone -- the property plain
    Historical VaR reacts to only slowly."""
    calm = _garch_returns(n=1500, seed=1)
    rng = np.random.default_rng(2)
    storm = np.concatenate([calm, rng.normal(0, 0.05, 250)])  # recent high-vol days

    calm_var = FilteredHistoricalSimulationVar(0.99, n_simulations=4000, seed=0).run(
        returns=calm
    ).value
    storm_var = FilteredHistoricalSimulationVar(0.99, n_simulations=4000, seed=0).run(
        returns=storm
    ).value
    assert storm_var > calm_var


def test_configurable_vol_model_runs():
    """A GJR-GARCH filter (o=1) is a supported configuration."""
    result = FilteredHistoricalSimulationVar(
        0.99, vol="Garch", o=1, dist="t", n_simulations=2000, seed=0
    ).run(returns=_garch_returns())
    assert result.value > 0


def test_rejects_bad_simulation_count():
    with pytest.raises(ValueError):
        FilteredHistoricalSimulationVar(0.99, n_simulations=0)


def test_close_to_historical_on_homoscedastic_data():
    """With constant volatility the filter has nothing to do, so FHS should land
    near plain Historical VaR."""
    returns = np.random.default_rng(9).normal(0, 0.02, 3000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fhs = FilteredHistoricalSimulationVar(0.99, n_simulations=8000, seed=0).run(
            returns=returns
        ).value
    plain = HistoricalVar(0.99).run(returns=returns).value
    assert fhs == pytest.approx(plain, rel=0.25)

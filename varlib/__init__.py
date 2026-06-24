"""
varlib — a readable, validated Value at Risk library.

Design goals
------------
1. One VaR method per file, with a clear name.
2. Every formula is written step by step. Every intermediate value is traced
   and returned, so the calculation can be inspected and audited line by line.
3. Models are validated, not just computed: the `backtest` subpackage provides
   industry-standard VaR backtests (Kupiec, Dynamic Quantile, Basel traffic light).
4. A small dependency surface: numpy and pandas for the engine, with scipy for
   the standard statistical functions (Normal, binomial, chi-square).

Quick start
-----------
>>> import numpy as np
>>> from varlib import HistoricalVar
>>> returns = np.random.default_rng(0).normal(0, 0.01, 1000)
>>> model = HistoricalVar(confidence=0.99)
>>> result = model.run(returns=returns)
>>> round(result.value, 4)  # the VaR  # doctest: +SKIP
0.0234
>>> round(result.expected_shortfall, 4)  # the ES, always >= VaR  # doctest: +SKIP
0.0269
>>> result.steps.keys()  # every intermediate is here  # doctest: +SKIP
dict_keys(['returns', 'losses', 'sorted_losses', 'var', 'es', ...])
"""

from varlib.base import VarModel, VarResult
from varlib.backtest.rolling import rolling_var, rolling_backtest
from varlib.report import BacktestReport, run_backtest
from varlib.models.historical import HistoricalVar, historical_var, historical_es
from varlib.models.historical_bootstrap import (
    HistoricalBootstrapVar,
    historical_bootstrap_var,
    historical_bootstrap_var_es,
)
from varlib.models.parametric_brownian import (
    ParametricBrownianVar,
    parametric_brownian_var,
    parametric_brownian_var_es,
)
from varlib.models.parametric_ou import (
    ParametricOuVar,
    parametric_ou_var,
    parametric_ou_var_es,
)
from varlib.models.parametric_jump import (
    ParametricJumpVar,
    parametric_jump_var,
    parametric_jump_var_es,
)
from varlib.models.ewma import (
    EwmaVar,
    ewma_var,
    ewma_var_es,
)

__all__ = [
    "VarModel",
    "VarResult",
    "rolling_var",
    "rolling_backtest",
    "BacktestReport",
    "run_backtest",
    "HistoricalVar",
    "historical_var",
    "historical_es",
    "HistoricalBootstrapVar",
    "historical_bootstrap_var",
    "historical_bootstrap_var_es",
    "ParametricBrownianVar",
    "parametric_brownian_var",
    "parametric_brownian_var_es",
    "ParametricOuVar",
    "parametric_ou_var",
    "parametric_ou_var_es",
    "ParametricJumpVar",
    "parametric_jump_var",
    "parametric_jump_var_es",
    "EwmaVar",
    "ewma_var",
    "ewma_var_es",
]

__version__ = "0.1.0"

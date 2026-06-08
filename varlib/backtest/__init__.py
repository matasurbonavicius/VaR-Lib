"""
VaR backtesting -- the part that turns a number into a validated number.

A VaR estimate is only useful if it is the right size: not so loose that it
never gets breached (wasting capital) and not so tight that it is breached far
too often (understating risk). Backtesting checks this against realised data.

The standard workflow on a single instrument (``rolling_backtest`` does steps
1-3 for you):

  1. Roll the VaR model through history, producing a VaR forecast for each day.
  2. Compare each forecast against the realised loss the next day.
  3. A "breach" (or "exception") is a day where the realised loss exceeded the
     forecast VaR.
  4. Run statistical tests on the breach sequence:
       - Kupiec POF:     are there roughly the right NUMBER of breaches?
       - Christoffersen: are the breaches INDEPENDENT, or do they cluster?
       - Dynamic Quantile: can the breaches be PREDICTED from their own past or
                         the VaR level? (Engle-Manganelli, the modern test.)
       - Traffic light:  Basel's supervisory green/yellow/red zoning.

Every test returns a small, named result object so the verdict and the numbers
behind it are both visible.
"""

from varlib.backtest.rolling import rolling_var, rolling_backtest
from varlib.backtest.kupiec import KupiecResult, kupiec_pof_test
from varlib.backtest.christoffersen import ChristoffersenResult, christoffersen_test
from varlib.backtest.dynamic_quantile import (
    DynamicQuantileResult,
    dynamic_quantile_test,
)
from varlib.backtest.traffic_light import (
    BreachSummary,
    TrafficLightResult,
    basel_traffic_light,
    breach_count,
    count_breaches,
)

__all__ = [
    "rolling_var",
    "rolling_backtest",
    "KupiecResult",
    "kupiec_pof_test",
    "ChristoffersenResult",
    "christoffersen_test",
    "DynamicQuantileResult",
    "dynamic_quantile_test",
    "BreachSummary",
    "TrafficLightResult",
    "basel_traffic_light",
    "breach_count",
    "count_breaches",
]

"""
EWMA / RiskMetrics VaR.

Implements J.P. Morgan/Reuters, *RiskMetrics(TM) -- Technical Document*, 4th ed.
(1996), included alongside this package as
``JPMorgan_RiskMetrics_TechnicalDocument.pdf``. See ``ewma.py`` for the
equation-by-equation mapping to the document's printed pages and the noted
departures from the literal model.
"""

from varlib.models.parametric.ewma.ewma import EwmaVar, ewma_var, ewma_var_es

__all__ = ["EwmaVar", "ewma_var", "ewma_var_es"]

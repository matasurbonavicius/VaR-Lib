## SEMI-PARAMETRIC VaR MODELS

1. Filtered Historical Simulation (FHS)

What makes a model semi-parametric is that it sits between the two extremes.
A non-parametric model assumes nothing about the return distribution and reads
risk straight off the empirical returns; a parametric model assumes a full
distribution (Normal, Student-t, a jump mixture) and reads risk off its formula.
A semi-parametric model does *both*: it fits a parametric model for one part of
the problem and stays empirical for the rest.

Filtered Historical Simulation is the canonical example. It keeps the tail fully
empirical -- every simulated shock is a real historical standardized residual,
nothing about the *shape* of the tail is assumed -- but it filters those shocks
through a parametric volatility model (GARCH). Parametric dynamics, empirical
shocks.

This buys the one thing plain historical simulation cannot do: react to *today's*
volatility. Historical simulation resamples raw past returns and so mixes calm-
and crisis-regime days into one pot, reacting to the current regime only slowly
as extreme days enter and leave the window. FHS divides each return by its own
conditional volatility before resampling, then re-inflates the drawn shocks by
the volatility *forecast* -- so the tail's shape is historical but its scale is
current.

For more information about each model, refer to the files and docstrings.

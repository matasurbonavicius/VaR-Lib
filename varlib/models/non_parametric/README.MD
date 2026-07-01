## NONPARAMETRIC VaR MODELS

1. Historical Simulation
2. Time Weighted Historical Simulation
3. Historical Bootstrap

What makes those models nonparametric is the fact that none of them try to model
a distribution. Each one of these models derive the final distribution
based on the empirical (observed) returns

These three models are the holy trinity and really all there is to 
nonparametric approaches. 

While simple, they all suffer from assuming that the returns in the future will
look like the returns in the past, which is not accurate. Additionally, 
both the Historical Bootstrap and the Historical Simulation suffer from the so called
plateu - which is when the large price movements in the past get excluded from the sample
period and the VaR decreases drastically just because of the sample window excluding a
single or more dominating returns. 

Time Weighted Historical Simulation tackles this issue
by weighting the latest data more than the past, therefore smoothing the plateu.

For more information about each model, refer to the files and docstrings.
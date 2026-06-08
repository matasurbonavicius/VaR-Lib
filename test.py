import numpy as np
from varlib import HistoricalVar

prices = np.array([100, 101, 99, 102, 98, 100, 103, 97])
model = HistoricalVar(confidence=0.99)
result = model.run(prices=prices)        # or run(returns=...)

print(result.value)                      # the VaR, as a positive loss fraction
print(result.expected_shortfall)         # the ES, always >= VaR
print(result.explain())                  # full step-by-step trace (VaR and ES)
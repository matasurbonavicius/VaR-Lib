"""
Chi-square upper-tail probability, computed without scipy.

Both the Kupiec and Dynamic Quantile tests produce a statistic that is
asymptotically chi-square distributed. To turn that statistic into a p-value we
need the chi-square survival function (1 - CDF).

We build it from the regularised upper incomplete gamma function Q(s, x), since
for `df` degrees of freedom:

    P(X > x) = Q(df/2, x/2)

Q is evaluated with the standard pairing of a series expansion (good for small
x) and a continued fraction (good for large x), following Numerical Recipes.
These are textbook, well-conditioned routines and need only `math`.
"""

from __future__ import annotations

import math

_MAX_ITER = 200
_EPS = 1e-12


def chi_square_sf(x: float, df: int) -> float:
    """Upper-tail probability P(X > x) for a chi-square with `df` dof."""
    if df < 1:
        raise ValueError("degrees of freedom must be >= 1")
    if x <= 0:
        return 1.0
    return _gamma_q(df / 2.0, x / 2.0)


def _gamma_q(s: float, x: float) -> float:
    """Regularised upper incomplete gamma function Q(s, x)."""
    if x < s + 1.0:
        # Series expansion converges fast here; Q = 1 - P.
        return 1.0 - _gamma_p_series(s, x)
    # Continued fraction converges fast in this region.
    return _gamma_q_continued_fraction(s, x)


def _gamma_p_series(s: float, x: float) -> float:
    """Lower regularised incomplete gamma P(s, x) via series expansion."""
    ln_gamma_s = math.lgamma(s)
    term = 1.0 / s
    total = term
    n = s
    for _ in range(_MAX_ITER):
        n += 1.0
        term *= x / n
        total += term
        if abs(term) < abs(total) * _EPS:
            break
    return total * math.exp(-x + s * math.log(x) - ln_gamma_s)


def _gamma_q_continued_fraction(s: float, x: float) -> float:
    """Upper regularised incomplete gamma Q(s, x) via continued fraction."""
    ln_gamma_s = math.lgamma(s)
    tiny = 1e-300
    b = x + 1.0 - s
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - s)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + s * math.log(x) - ln_gamma_s) * h

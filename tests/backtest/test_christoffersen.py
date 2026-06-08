"""Tests for the Christoffersen independence / conditional-coverage tests."""

import numpy as np
import pytest

from varlib.backtest import christoffersen_test


def test_independent_breaches_not_rejected():
    # Spread breaches evenly so there is no clustering.
    breaches = np.zeros(1000)
    breaches[::100] = 1.0  # every 100th day -> 10 breaches, well separated
    result = christoffersen_test(breaches, confidence=0.99)
    assert not result.reject_independence


def test_clustered_breaches_flagged_by_independence_test():
    # All breaches in one consecutive block -> strong dependence.
    breaches = np.zeros(1000)
    breaches[200:215] = 1.0  # 15 breaches back to back
    result = christoffersen_test(breaches, confidence=0.99)
    assert result.reject_independence
    assert result.p_value_independence < 0.05


def test_conditional_is_sum_of_components():
    breaches = np.zeros(1000)
    breaches[100:106] = 1.0
    result = christoffersen_test(breaches, 0.99)
    assert result.lr_conditional == pytest.approx(
        result.lr_unconditional + result.lr_independence
    )


def test_transition_counts_traced():
    breaches = np.array([0, 0, 1, 1, 0, 1, 0, 0], dtype=float)
    result = christoffersen_test(breaches, 0.99)
    t = result.steps["transitions"]
    # Manually: pairs (0,0),(0,1),(1,1),(1,0),(0,1),(1,0),(0,0)
    assert t["n00"] == 2
    assert t["n01"] == 2
    assert t["n10"] == 2
    assert t["n11"] == 1


def test_requires_two_observations():
    with pytest.raises(ValueError):
        christoffersen_test(np.array([1.0]), 0.99)

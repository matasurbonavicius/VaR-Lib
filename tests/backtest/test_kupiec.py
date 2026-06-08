"""Tests for the Kupiec POF test and its chi-square machinery."""

import numpy as np
import pytest

from varlib.backtest import kupiec_pof_test
from varlib.backtest._chi_square import chi_square_sf


def test_chi_square_sf_known_values():
    # Median of chi-square(1) is ~0.4549; sf there is ~0.5.
    assert chi_square_sf(0.4549, 1) == pytest.approx(0.5, abs=1e-3)
    # 3.841 is the 95th percentile of chi-square(1): sf ~ 0.05.
    assert chi_square_sf(3.841, 1) == pytest.approx(0.05, abs=1e-3)
    # 5.991 is the 95th percentile of chi-square(2): sf ~ 0.05.
    assert chi_square_sf(5.991, 2) == pytest.approx(0.05, abs=1e-3)


def test_correct_breach_rate_is_not_rejected():
    # Exactly 1% breaches over 1000 days at 99% -> model should pass.
    breaches = np.zeros(1000)
    breaches[:10] = 1.0
    result = kupiec_pof_test(breaches, confidence=0.99)
    assert result.n_breaches == 10
    assert not result.reject_at_5pct
    assert result.p_value > 0.05


def test_far_too_many_breaches_is_rejected():
    # 50 breaches over 1000 days at 99% (5x too many) -> reject.
    breaches = np.zeros(1000)
    breaches[:50] = 1.0
    result = kupiec_pof_test(breaches, confidence=0.99)
    assert result.reject_at_5pct
    assert result.p_value < 0.05


def test_zero_breaches_handled():
    breaches = np.zeros(500)
    result = kupiec_pof_test(breaches, confidence=0.99)
    assert result.n_breaches == 0
    assert np.isfinite(result.statistic)


def test_traces_likelihoods():
    breaches = np.zeros(1000)
    breaches[:12] = 1.0
    result = kupiec_pof_test(breaches, 0.99)
    assert {"log_lik_null", "log_lik_alt", "statistic", "p_value"} <= set(result.steps)

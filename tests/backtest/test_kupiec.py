"""Tests for the Kupiec POF test."""

import numpy as np

from varlib.backtest import kupiec_pof_test


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

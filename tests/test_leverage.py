"""Leverage analysis must refuse to flatter risk-taking."""

import numpy as np
import pandas as pd
import pytest

from trader.leverage import (
    analyse,
    leverage_table,
    outcome_distribution,
    zero_edge_control,
)


def _series(n=1500, mu=0.0005, sd=0.011, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.Series(rng.normal(mu, sd, n), index=idx)


def test_growth_peaks_at_kelly_and_falls_after():
    """The central claim: past Kelly you get more risk for less money."""
    p = analyse(_series())
    k = p.kelly_leverage
    assert p.growth(k) > p.growth(k * 0.5)
    assert p.growth(k) > p.growth(k * 1.5)
    assert p.growth(k) > p.growth(k * 2)


def test_growth_at_twice_kelly_is_zero():
    p = analyse(_series())
    assert p.growth(2 * p.kelly_leverage) == pytest.approx(0.0, abs=1e-9)


def test_growth_at_kelly_equals_half_sharpe_squared():
    p = analyse(_series())
    assert p.growth(p.kelly_leverage) == pytest.approx(p.sharpe**2 / 2, rel=1e-9)


def test_unproven_edge_yields_zero_recommended_leverage():
    """A Sharpe whose CI spans zero must produce a recommendation of zero."""
    p = analyse(_series(n=300, mu=0.0002, sd=0.02))
    assert not p.edge_is_established
    assert p.recommended_leverage() == 0.0


def test_established_edge_yields_fractional_kelly():
    p = analyse(_series(n=4000, mu=0.0012, sd=0.008))
    assert p.edge_is_established
    rec = p.recommended_leverage(kelly_fraction=0.25)
    assert rec == pytest.approx(p.kelly_leverage * 0.25)
    assert rec < p.kelly_leverage


def test_standard_error_shrinks_with_more_data():
    short = analyse(_series(n=400, seed=1))
    long = analyse(_series(n=4000, seed=1))
    assert long.sharpe_stderr < short.sharpe_stderr


def test_leverage_raises_both_upside_and_ruin():
    """The honest tradeoff: more leverage buys upside AND downside together."""
    returns = _series(n=1200)
    low = outcome_distribution(returns, 1.0, num_paths=3000)
    high = outcome_distribution(returns, 4.0, num_paths=3000)
    assert high["p_double"] > low["p_double"]
    assert high["p_lose_half"] > low["p_lose_half"]


def test_zero_edge_control_removes_the_edge_not_the_risk():
    returns = _series(n=1200, mu=0.001)
    control = zero_edge_control(returns)
    assert control.mean() == pytest.approx(0.0, abs=1e-15)
    assert control.std() == pytest.approx(returns.std(), rel=1e-9)


def test_leverage_still_offers_upside_with_no_edge():
    """The seduction: leverage buys lottery tickets even when EV is negative.

    This is why 'higher chance of a big win' is not evidence of a good strategy.
    """
    control = zero_edge_control(_series(n=1200, mu=0.0))
    high = outcome_distribution(control, 4.0, num_paths=4000)
    assert high["p_double"] > 0.02
    assert high["mean_log_growth"] < 0.0


def test_leverage_table_covers_requested_levels():
    table = leverage_table(_series(n=800), levels=[1.0, 2.0, 3.0], num_paths=800)
    assert list(table["leverage"]) == [1.0, 2.0, 3.0]
    assert {"p_double", "p_ruin", "median"}.issubset(table.columns)


def test_short_series_is_rejected():
    with pytest.raises(ValueError, match="at least 30"):
        analyse(_series(n=10))

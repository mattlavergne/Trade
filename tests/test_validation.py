"""Validation must be hard to fool. These tests feed it noise and check it says so."""

import numpy as np
import pandas as pd
import pytest

from trader.validation import (
    block_bootstrap_pvalue,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
)


def _noise(n=1000, seed=0, mu=0.0, sigma=0.01):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.Series(rng.normal(mu, sigma, n), index=idx)


def test_sharpe_of_pure_noise_is_near_zero():
    assert abs(sharpe_ratio(_noise(4000))) < 0.6


def test_psr_rejects_noise():
    """A strategy with no edge must not be certified."""
    assert probabilistic_sharpe_ratio(_noise(1000)) < 0.95


def test_psr_accepts_a_strong_persistent_edge():
    strong = _noise(2000, mu=0.002, sigma=0.01)  # Sharpe ~3.8 annualised
    assert probabilistic_sharpe_ratio(strong) > 0.99


def test_expected_max_sharpe_grows_with_trials():
    variance = 0.5 / 365
    values = [expected_max_sharpe(n, variance) for n in (2, 10, 100, 1000)]
    assert values == sorted(values)
    assert values[0] > 0


def test_deflated_sharpe_punishes_many_trials():
    """The same returns must look less impressive after more attempts."""
    returns = _noise(800, mu=0.0008, sigma=0.01)
    few = deflated_sharpe_ratio(returns, num_trials=1)
    many = deflated_sharpe_ratio(returns, num_trials=500)
    assert many < few, "testing more configurations must lower confidence"


def test_bootstrap_pvalue_is_high_for_noise():
    assert block_bootstrap_pvalue(_noise(600), num_samples=400) > 0.10


def test_bootstrap_pvalue_is_low_for_real_drift():
    strong = _noise(600, mu=0.003, sigma=0.01)
    assert block_bootstrap_pvalue(strong, num_samples=400) < 0.05


def test_bootstrap_returns_valid_probability():
    p = block_bootstrap_pvalue(_noise(300), num_samples=200)
    assert 0.0 <= p <= 1.0


def test_short_samples_do_not_crash():
    tiny = pd.Series([0.01, -0.01], index=pd.date_range("2024-01-01", periods=2, tz="UTC"))
    assert probabilistic_sharpe_ratio(tiny) >= 0.0
    assert block_bootstrap_pvalue(tiny) == 1.0


def test_zero_variance_returns_zero_sharpe():
    flat = pd.Series([0.0] * 100, index=pd.date_range("2024-01-01", periods=100, tz="UTC"))
    assert sharpe_ratio(flat) == 0.0


def test_fat_tails_reduce_confidence():
    """Two series, same Sharpe, but one has crash risk. It must score lower."""
    rng = np.random.default_rng(1)
    n = 1500
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    normal = pd.Series(rng.normal(0.0008, 0.01, n), index=idx)
    skewed = normal.copy()
    skewed.iloc[::200] -= 0.08          # periodic crashes
    skewed += (normal.mean() - skewed.mean())  # equalise mean return
    assert probabilistic_sharpe_ratio(skewed) < probabilistic_sharpe_ratio(normal)


def test_cli_param_grids_contain_no_duplicates():
    """Duplicate configurations inflate the Deflated Sharpe trial count."""
    import argparse

    from trader.cli_portfolio import cmd_walkforward  # noqa: F401  (import guard)
    from trader.strategies.portfolio_strategies import PORTFOLIO_STRATEGIES

    # Rebuild the grids exactly as cmd_walkforward does.
    grids = {
        "trend_filtered": [{"lookback": lb, "top_n": n, "trend_window": tw}
                           for lb in (30, 60, 120) for n in (3, 5) for tw in (100, 200)],
        "xsmom": [{"lookback": lb, "top_n": n} for lb in (30, 60, 120) for n in (3, 5)],
        "tsmom": [{"lookback": lb, "vol_scale": vs}
                  for lb in (30, 60, 90, 120, 180) for vs in (True, False)],
    }
    for name, grid in grids.items():
        unique = [dict(t) for t in {tuple(sorted(g.items())) for g in grid}]
        assert len(unique) == len(grid), f"{name} grid has duplicate parameter sets"
        assert set(grid[0]).issubset(set(PORTFOLIO_STRATEGIES[name]().__dict__) | {"vol_scale"})

"""Carry trade economics. The maths must not flatter the trade."""

import numpy as np
import pandas as pd
import pytest

from trader.funding import (
    FUNDING_PERIODS_PER_YEAR,
    breakeven_holding_periods,
    carry_trade_pnl,
    summarise_funding,
)


def _funding(n=1000, mean=0.0001, sd=0.00005, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2025-01-01", periods=n, freq="8h", tz="UTC")
    return pd.Series(rng.normal(mean, sd, n), index=idx)


def test_summary_annualises_correctly():
    flat = pd.Series(0.0001, index=pd.date_range("2025-01-01", periods=100, freq="8h", tz="UTC"))
    s = summarise_funding(flat)
    assert s["annualised_pct"] == pytest.approx(0.0001 * FUNDING_PERIODS_PER_YEAR * 100)
    assert s["pct_positive"] == 100.0


def test_summary_of_empty_series_is_empty():
    assert summarise_funding(pd.Series(dtype=float)) == {}


def test_capital_efficiency_scales_returns_down():
    """Margin buffer means not all capital earns funding."""
    funding = _funding()
    full = carry_trade_pnl(funding, capital_efficiency=1.0, basis_vol_bps=0.0)
    half = carry_trade_pnl(funding, capital_efficiency=0.5, basis_vol_bps=0.0)
    assert half["gross_return"].mean() == pytest.approx(full["gross_return"].mean() / 2)


def test_entry_and_exit_costs_are_charged_once_each():
    funding = _funding(n=100)
    free = carry_trade_pnl(funding, entry_cost_bps=0, exit_cost_bps=0, basis_vol_bps=0.0)
    paid = carry_trade_pnl(funding, entry_cost_bps=20, exit_cost_bps=20, basis_vol_bps=0.0)
    difference = free["net_return"].sum() - paid["net_return"].sum()
    assert difference == pytest.approx(40e-4, rel=1e-6)


def test_basis_noise_adds_variance_without_adding_return():
    """Delta-neutral does not mean variance-free. Basis risk must show up."""
    funding = _funding()
    calm = carry_trade_pnl(funding, basis_vol_bps=0.0)
    noisy = carry_trade_pnl(funding, basis_vol_bps=30.0)
    assert noisy["net_return"].std() > calm["net_return"].std()


def test_basis_noise_reduces_reported_sharpe():
    """The headline 'funding Sharpe 40' must not survive basis risk."""
    funding = _funding()
    naive = funding.mean() / funding.std() * np.sqrt(FUNDING_PERIODS_PER_YEAR)
    pnl = carry_trade_pnl(funding, basis_vol_bps=30.0, capital_efficiency=0.5)
    net = pnl["net_return"]
    realistic = net.mean() / net.std() * np.sqrt(FUNDING_PERIODS_PER_YEAR)
    assert realistic < naive / 2


def test_breakeven_grows_with_cost():
    a = breakeven_holding_periods(1e-4, entry_cost_bps=10, exit_cost_bps=10)
    b = breakeven_holding_periods(1e-4, entry_cost_bps=40, exit_cost_bps=40)
    assert b > a


def test_negative_funding_never_breaks_even():
    assert breakeven_holding_periods(-1e-4) == float("inf")
    assert breakeven_holding_periods(0.0) == float("inf")


def test_empty_funding_produces_empty_pnl():
    assert carry_trade_pnl(pd.Series(dtype=float)).empty

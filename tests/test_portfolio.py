"""Portfolio engine tests, including the multi-asset look-ahead guarantee."""

import numpy as np
import pandas as pd
import pytest

from trader.costs import get_venue
from trader.portfolio import PortfolioStrategy, run_portfolio_backtest
from trader.risk import RiskConfig
from trader.universe import Panel


def _panel(n=400, symbols=("A", "B", "C"), seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    frames = {}
    for i, sym in enumerate(symbols):
        rets = rng.normal(drift, 0.03, n)
        close = pd.Series(100.0 * np.exp(np.cumsum(rets)), index=index)
        frames[sym] = pd.DataFrame({
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": 1000.0,
        }, index=index)
    return Panel(frames)


class Flat(PortfolioStrategy):
    name = "flat"

    def signals(self, panel):
        return pd.DataFrame(0.0, index=panel.index, columns=panel.symbols)


class AlwaysLong(PortfolioStrategy):
    name = "long"

    def signals(self, panel):
        return pd.DataFrame(1.0, index=panel.index, columns=panel.symbols)


class Oracle(PortfolioStrategy):
    """Cheats: signals exactly tomorrow's return. Must NOT profit."""

    name = "oracle"

    def signals(self, panel):
        future = panel.close.pct_change().shift(-1)
        return np.sign(future).fillna(0.0)


def test_flat_strategy_preserves_capital_exactly():
    result = run_portfolio_backtest(_panel(), Flat(), get_venue("kraken"),
                                    starting_cash=10_000.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(10_000.0)
    assert result.total_cost == 0.0
    assert result.turnover.sum() == 0.0


def test_oracle_cannot_exploit_next_day_knowledge():
    """The engine's one-bar lag must neutralise a perfect forecast.

    An oracle signalling tomorrow's direction is executed a day late, so it
    should NOT produce spectacular returns. If this test fails, every
    portfolio result in the repo is invalid.
    """
    panel = _panel(n=600, seed=3)
    result = run_portfolio_backtest(panel, Oracle(), get_venue("frictionless"),
                                    starting_cash=10_000.0)
    growth = result.equity_curve.iloc[-1] / 10_000.0
    assert growth < 3.0, (
        f"LOOK-AHEAD BIAS: oracle grew {growth:.1f}x. A one-bar-lagged perfect "
        "forecast must not compound like this."
    )


def test_costs_scale_with_turnover():
    panel = _panel(n=500, seed=7)
    cheap = run_portfolio_backtest(panel, AlwaysLong(), get_venue("frictionless"),
                                   starting_cash=10_000.0)
    dear = run_portfolio_backtest(panel, AlwaysLong(), get_venue("coinbase"),
                                  starting_cash=10_000.0)
    assert dear.total_cost > cheap.total_cost
    assert cheap.total_cost == 0.0


def test_wider_band_reduces_turnover_end_to_end():
    panel = _panel(n=500, seed=11)
    tight = run_portfolio_backtest(panel, AlwaysLong(), get_venue("kraken"),
                                   config=RiskConfig(no_trade_band=0.0))
    wide = run_portfolio_backtest(panel, AlwaysLong(), get_venue("kraken"),
                                  config=RiskConfig(no_trade_band=0.75))
    assert wide.turnover.sum() < tight.turnover.sum()


def test_leverage_cap_holds_across_whole_backtest():
    panel = _panel(n=400, seed=5)
    cfg = RiskConfig(target_volatility=2.0, max_leverage=1.0)
    result = run_portfolio_backtest(panel, AlwaysLong(), get_venue("kraken"), config=cfg)
    assert result.exposure.max() <= 1.0 + 1e-6


def test_drawdown_stop_engages():
    """A relentlessly falling market must end with exposure cut to zero."""
    n = 500
    index = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC")
    close = pd.Series(100.0 * np.exp(np.cumsum(np.full(n, -0.01))), index=index)
    frames = {s: pd.DataFrame({"open": close, "high": close, "low": close,
                               "close": close, "volume": 1.0}, index=index)
              for s in ("A", "B")}
    result = run_portfolio_backtest(Panel(frames), AlwaysLong(), get_venue("kraken"),
                                    config=RiskConfig(max_drawdown_stop=0.25,
                                                      drawdown_scale_start=0.10))
    assert result.exposure.iloc[-1] == pytest.approx(0.0, abs=1e-6)


def test_signals_outside_unit_range_are_rejected():
    class TooBig(PortfolioStrategy):
        name = "toobig"

        def signals(self, panel):
            return pd.DataFrame(2.0, index=panel.index, columns=panel.symbols)

    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        run_portfolio_backtest(_panel(), TooBig(), get_venue("kraken"))


def test_misaligned_signals_are_rejected():
    class Misaligned(PortfolioStrategy):
        name = "mis"

        def signals(self, panel):
            return pd.DataFrame(0.0, index=panel.index[:-10], columns=panel.symbols)

    with pytest.raises(ValueError, match="align"):
        run_portfolio_backtest(_panel(), Misaligned(), get_venue("kraken"))


def test_warmup_period_does_not_trade():
    result = run_portfolio_backtest(_panel(n=300), AlwaysLong(), get_venue("kraken"),
                                    warmup=100)
    assert result.turnover.iloc[:100].sum() == 0.0


def test_vol_targeting_reduces_realised_volatility():
    """The core claim: targeting 20% vol on 48% vol assets must dampen it."""
    panel = _panel(n=800, seed=13)
    result = run_portfolio_backtest(panel, AlwaysLong(), get_venue("frictionless"),
                                    config=RiskConfig(target_volatility=0.20))
    realised = result.daily_returns().std() * np.sqrt(365)
    asset_vol = panel.returns().mean(axis=1).std() * np.sqrt(365)
    assert realised < asset_vol

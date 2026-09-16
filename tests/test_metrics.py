"""Metrics must be computed net of costs and must flag their own weakness."""

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from trader.broker import PaperBroker, Side
from trader.costs import get_venue
from trader.metrics import MIN_TRADES_FOR_SIGNIFICANCE, _round_trip_pnls, compute

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _curve(values):
    index = pd.date_range("2025-01-01", periods=len(values), freq="1D", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def test_max_drawdown_matches_hand_calculation():
    m = compute(_curve([100, 120, 60, 80]), [], [], timeframe="1d",
                total_fees=0.0, total_slippage=0.0)
    assert m.max_drawdown_pct == pytest.approx(-50.0)  # 120 -> 60


def test_total_return_matches_endpoints():
    m = compute(_curve([50, 75]), [], [], timeframe="1d",
                total_fees=0.0, total_slippage=0.0)
    assert m.total_return_pct == pytest.approx(50.0)


def test_low_trade_count_is_flagged_as_noise():
    m = compute(_curve([50, 51, 52]), [], [], timeframe="1d",
                total_fees=0.0, total_slippage=0.0)
    assert any("round trips" in w for w in m.warnings)


def test_heavy_fee_drag_is_flagged():
    m = compute(_curve([50, 45]), [], [], timeframe="1d",
                total_fees=8.0, total_slippage=0.5)
    assert m.fee_drag_pct == pytest.approx(17.0)
    assert any("Costs consumed" in w for w in m.warnings)


def test_rejections_are_surfaced_as_a_warning():
    m = compute(_curve([50, 50]), [], ["a rejection"], timeframe="1d",
                total_fees=0.0, total_slippage=0.0)
    assert any("rejected" in w for w in m.warnings)


def test_round_trip_pnl_is_net_of_fees():
    """A flat-price round trip must show NEGATIVE pnl once fees are paid."""
    broker = PaperBroker(starting_cash=10_000.0, venue=get_venue("coinbase"))
    broker.execute(NOW, Side.BUY, 1.0, 100.0)
    broker.execute(NOW, Side.SELL, 1.0, 100.0)
    pnls = _round_trip_pnls(broker.fills)
    assert len(pnls) == 1
    assert pnls[0] < 0, "flat-price round trip must lose the fees, not break even"


def test_fifo_pairing_splits_partial_exits():
    broker = PaperBroker(starting_cash=100_000.0, venue=get_venue("frictionless"))
    broker.execute(NOW, Side.BUY, 2.0, 100.0)
    broker.execute(NOW, Side.SELL, 1.0, 110.0)
    broker.execute(NOW, Side.SELL, 1.0, 90.0)
    pnls = _round_trip_pnls(broker.fills)
    assert pnls == pytest.approx([10.0, -10.0])


def test_short_sample_is_flagged():
    m = compute(_curve([50] * 10), [], [], timeframe="1d",
                total_fees=0.0, total_slippage=0.0)
    assert any("days" in w for w in m.warnings)


def test_insignificant_edge_is_flagged_when_trade_count_is_adequate():
    """Many trades with a near-zero mean must be called out as noise."""
    rng = np.random.default_rng(0)
    broker = PaperBroker(starting_cash=1_000_000.0, venue=get_venue("frictionless"))
    for _ in range(MIN_TRADES_FOR_SIGNIFICANCE + 20):
        entry = 100.0
        exit_price = 100.0 + rng.normal(0.0, 5.0)
        broker.execute(NOW, Side.BUY, 1.0, entry)
        broker.execute(NOW, Side.SELL, 1.0, max(exit_price, 1.0))
    m = compute(_curve([50, 50.1]), broker.fills, [], timeframe="1d",
                total_fees=0.0, total_slippage=0.0)
    assert any("t-statistic" in w for w in m.warnings)

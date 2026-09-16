"""The broker must charge honestly and refuse impossible orders."""

from datetime import datetime, timezone

import pytest

from trader.broker import PaperBroker, RejectReason, Side
from trader.costs import VenueCosts, get_venue

NOW = datetime(2025, 1, 1, tzinfo=timezone.utc)


def test_round_trip_at_unchanged_price_loses_exactly_the_costs():
    broker = PaperBroker(starting_cash=50.0, venue=get_venue("coinbase"))
    broker.target_position_value(NOW, 50.0, 100.0)
    broker.liquidate(NOW, 100.0)
    expected = 50.0 - broker.total_fees - broker.total_slippage
    assert broker.equity(100.0) == pytest.approx(expected, abs=1e-6)
    assert broker.equity(100.0) < 50.0, "a round trip at a flat price must lose money"


def test_orders_below_min_notional_are_rejected_not_shrunk():
    venue = VenueCosts(name="t", maker_bps=0, taker_bps=0, min_notional_usd=10.0)
    broker = PaperBroker(starting_cash=50.0, venue=venue)
    fill = broker.execute(NOW, Side.BUY, 0.05, 100.0)  # $5 notional, below $10 min
    assert fill is None
    assert broker.position == 0.0
    assert broker.rejections[-1].reason is RejectReason.BELOW_MIN_NOTIONAL


def test_cannot_spend_more_cash_than_held():
    broker = PaperBroker(starting_cash=50.0, venue=get_venue("kraken"))
    fill = broker.execute(NOW, Side.BUY, 10.0, 100.0)  # $1000 order on $50
    assert fill is None
    assert broker.rejections[-1].reason is RejectReason.INSUFFICIENT_CASH
    assert broker.cash == 50.0


def test_cannot_sell_more_than_position():
    broker = PaperBroker(starting_cash=50.0, venue=get_venue("kraken"))
    fill = broker.execute(NOW, Side.SELL, 1.0, 100.0)
    assert fill is None
    assert broker.rejections[-1].reason is RejectReason.INSUFFICIENT_POSITION


def test_no_implicit_leverage_after_many_buys():
    broker = PaperBroker(starting_cash=50.0, venue=get_venue("kraken"))
    for _ in range(20):
        broker.target_position_value(NOW, 1e9, 100.0)
    assert broker.cash >= -1e-9, "cash must never go negative"
    assert broker.position_value(100.0) <= 50.0 + 1e-6


def test_slippage_always_hurts_in_the_direction_of_the_trade():
    broker = PaperBroker(starting_cash=1000.0, venue=get_venue("kraken"))
    buy = broker.execute(NOW, Side.BUY, 1.0, 100.0)
    sell = broker.execute(NOW, Side.SELL, 1.0, 100.0)
    assert buy.fill_price > 100.0, "buys must fill above the reference price"
    assert sell.fill_price < 100.0, "sells must fill below the reference price"


def test_fee_matches_venue_schedule():
    venue = get_venue("coinbase")
    broker = PaperBroker(starting_cash=10_000.0, venue=venue)
    fill = broker.execute(NOW, Side.BUY, 1.0, 100.0)
    assert fill.fee == pytest.approx(fill.notional * venue.taker_bps * 1e-4)


def test_frictionless_venue_is_actually_free():
    broker = PaperBroker(starting_cash=50.0, venue=get_venue("frictionless"))
    broker.target_position_value(NOW, 50.0, 100.0)
    broker.liquidate(NOW, 100.0)
    assert broker.equity(100.0) == pytest.approx(50.0, abs=1e-9)

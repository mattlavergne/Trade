"""Cost model sanity. These numbers drive every conclusion the harness draws."""

import pytest

from trader.costs import VENUES, VenueCosts, breakeven_move_pct, get_venue


def test_all_venues_have_non_negative_costs():
    for name, venue in VENUES.items():
        assert venue.taker_bps >= 0, name
        assert venue.maker_bps >= 0, name
        assert venue.min_notional_usd >= 0, name


def test_taker_is_never_cheaper_than_maker():
    for name, venue in VENUES.items():
        assert venue.taker_bps >= venue.maker_bps, f"{name} taker < maker is implausible"


def test_round_trip_is_two_one_way_costs():
    venue = VenueCosts(name="t", maker_bps=10.0, taker_bps=20.0,
                       min_notional_usd=0.0, half_spread_bps=5.0, impact_bps_per_1k=0.0)
    assert venue.round_trip_bps(is_taker=True) == pytest.approx(50.0)
    assert venue.round_trip_bps(is_taker=False) == pytest.approx(20.0)


def test_maker_orders_pay_no_half_spread():
    venue = get_venue("coinbase")
    assert venue.slippage_bps(100.0, is_taker=False) == 0.0
    assert venue.slippage_bps(100.0, is_taker=True) > 0.0


def test_impact_scales_with_size():
    venue = get_venue("kraken")
    assert venue.slippage_bps(10_000.0, is_taker=True) > venue.slippage_bps(50.0, is_taker=True)


def test_commission_free_equities_beat_crypto_by_an_order_of_magnitude():
    """The central finding: venue choice dominates strategy choice at $50."""
    crypto = breakeven_move_pct(get_venue("coinbase"))
    equities = breakeven_move_pct(get_venue("alpaca_equities"))
    assert equities * 10 < crypto


def test_unknown_venue_raises_helpfully():
    with pytest.raises(KeyError, match="Unknown venue"):
        get_venue("definitely_not_a_venue")

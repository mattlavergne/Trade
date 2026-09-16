"""Trading cost models.

Costs are the whole ballgame at small account sizes. A strategy that looks
brilliant gross-of-fees is usually a fee-donation machine net-of-fees, and the
smaller the account the more true that gets. Every number here is a drag on
returns that must be modelled honestly, or the backtest is a lie.

IMPORTANT: the fee figures below are *defaults for the lowest volume tier* and
change over time and by account. Verify them against your own fee schedule
before trusting any backtest that uses them. Underestimating fees is the single
most common way a backtest flatters a losing strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

BPS = 1e-4  # one basis point


@dataclass(frozen=True)
class VenueCosts:
    """Cost and constraint profile for a single exchange.

    Attributes:
        name: Exchange identifier (matches the ccxt id).
        maker_bps: Maker fee in basis points (limit order that adds liquidity).
        taker_bps: Taker fee in basis points (order that crosses the spread).
        min_notional_usd: Smallest order the venue will accept, in USD. Orders
            below this are rejected outright -- the constraint that quietly
            kills most small-account strategies.
        half_spread_bps: Typical half-spread paid when crossing the book. A
            taker order pays this *on top of* the taker fee.
        impact_bps_per_1k: Extra slippage in bps per $1,000 of order notional,
            a crude linear market-impact model. Negligible at $50, included so
            the same harness stays honest if the account ever grows.
    """

    name: str
    maker_bps: float
    taker_bps: float
    min_notional_usd: float
    half_spread_bps: float = 1.0
    impact_bps_per_1k: float = 0.5

    def fee_bps(self, *, is_taker: bool) -> float:
        return self.taker_bps if is_taker else self.maker_bps

    def slippage_bps(self, notional_usd: float, *, is_taker: bool) -> float:
        """Slippage in bps for an order of the given size.

        Maker orders do not cross the spread, so they pay no half-spread. They
        pay something worse that this model cannot express: adverse selection
        and non-fills. See `PaperBroker` for how fills are gated.
        """
        if not is_taker:
            return 0.0
        return self.half_spread_bps + self.impact_bps_per_1k * (notional_usd / 1_000.0)

    def round_trip_bps(self, *, is_taker: bool = True) -> float:
        """Total cost in bps to enter and exit one position. The hurdle rate
        any strategy must clear before it has made a single cent."""
        one_way = self.fee_bps(is_taker=is_taker) + self.slippage_bps(0.0, is_taker=is_taker)
        return 2.0 * one_way


# Lowest-tier retail fee schedules for US-accessible venues.
# VERIFY THESE AGAINST YOUR ACCOUNT. They are the defaults, not a promise.
VENUES: dict[str, VenueCosts] = {
    # Coinbase Advanced Trade, <$10k 30-day volume. Brutally expensive at the
    # bottom tier -- a 1.20% taker fee means a 2.4% round trip.
    "coinbase": VenueCosts(
        name="coinbase",
        maker_bps=60.0,
        taker_bps=120.0,
        min_notional_usd=1.0,
        half_spread_bps=1.5,
    ),
    # Kraken Pro, <$10k 30-day volume. The cheapest of the three for a US retail
    # account, which is why it is the default here.
    "kraken": VenueCosts(
        name="kraken",
        maker_bps=25.0,
        taker_bps=40.0,
        min_notional_usd=1.0,
        half_spread_bps=1.0,
    ),
    # Binance.US base tier.
    "binanceus": VenueCosts(
        name="binanceus",
        maker_bps=40.0,
        taker_bps=60.0,
        min_notional_usd=10.0,
        half_spread_bps=1.0,
    ),
    # --- Commission-free US equities -------------------------------------
    # Alpaca charges no commission on US equities and supports fractional
    # shares, which removes the per-trade fee that annihilates small crypto
    # accounts. You still pay the bid-ask spread, and for a $50 account the
    # SEC/FINRA regulatory fees on sells round to fractions of a cent.
    #
    # The binding constraint here is NOT cost, it is the Pattern Day Trader
    # rule: under $25,000 equity you get 3 day trades per rolling 5 business
    # days. Strategies that round-trip intraday are simply not runnable.
    "alpaca_equities": VenueCosts(
        name="alpaca_equities",
        maker_bps=0.0,
        taker_bps=0.0,
        min_notional_usd=1.0,
        half_spread_bps=2.0,      # liquid large-cap ETF; wider for small caps
        impact_bps_per_1k=0.1,
    ),
    # Retail spot FX, for comparison only. Modelled as spread-only (most
    # retail FX brokers embed their fee in the spread rather than charging
    # commission). ~0.8 pip round trip on EUR/USD at ~1.08 is about 7.4bps.
    # Looks cheap. It is not: see README on why leverage, not spread, is what
    # destroys retail FX accounts.
    "retail_fx": VenueCosts(
        name="retail_fx",
        maker_bps=0.0,
        taker_bps=0.0,
        min_notional_usd=1.0,
        half_spread_bps=3.7,
        impact_bps_per_1k=0.0,
    ),
    # Zero-cost venue, for isolating strategy logic from cost drag while
    # debugging. NEVER quote a backtest run on this as a real result.
    "frictionless": VenueCosts(
        name="frictionless",
        maker_bps=0.0,
        taker_bps=0.0,
        min_notional_usd=0.0,
        half_spread_bps=0.0,
        impact_bps_per_1k=0.0,
    ),
}


def get_venue(name: str) -> VenueCosts:
    try:
        return VENUES[name]
    except KeyError:
        raise KeyError(
            f"Unknown venue {name!r}. Known venues: {sorted(VENUES)}"
        ) from None


def breakeven_move_pct(venue: VenueCosts, *, is_taker: bool = True) -> float:
    """How far price must move, in percent, for a round trip to break even.

    This is the number to stare at before writing a strategy. If your signal's
    average predicted move is smaller than this, no amount of cleverness in the
    code will save it.
    """
    return venue.round_trip_bps(is_taker=is_taker) * BPS * 100.0

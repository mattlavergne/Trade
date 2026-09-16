"""Paper broker: simulated order execution with realistic frictions.

Design rules, each one guarding against a way backtests lie:

1. Orders fill at the *next* bar's open, never the bar that generated the
   signal. Filling on the signal bar's close is look-ahead bias and is the
   single most common reason a backtest shows profit that evaporates live.
2. Every fill pays an explicit fee and slippage, recorded separately from PnL
   so fee drag is visible rather than buried.
3. Orders below the venue's minimum notional are REJECTED, not silently shrunk.
   At a $50 account this rejects a lot of trades. That is the point: it is real.
4. The broker cannot go cash-negative. No implicit leverage, no margin. If you
   want leverage you have to add it deliberately and own the consequences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from .costs import BPS, VenueCosts

log = logging.getLogger(__name__)


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class RejectReason(str, Enum):
    BELOW_MIN_NOTIONAL = "below_min_notional"
    INSUFFICIENT_CASH = "insufficient_cash"
    INSUFFICIENT_POSITION = "insufficient_position"
    NON_POSITIVE_SIZE = "non_positive_size"


@dataclass
class Fill:
    """A completed trade, with costs broken out for honest accounting."""

    timestamp: datetime
    side: Side
    quantity: float
    reference_price: float
    fill_price: float
    notional: float
    fee: float
    slippage_cost: float
    cash_after: float
    position_after: float

    @property
    def total_cost(self) -> float:
        return self.fee + self.slippage_cost


@dataclass
class Rejection:
    timestamp: datetime
    side: Side
    quantity: float
    reference_price: float
    reason: RejectReason
    detail: str


@dataclass
class PaperBroker:
    """Cash-and-one-asset paper trading account.

    Args:
        starting_cash: Initial USD balance.
        venue: Cost profile to charge against every fill.
        is_taker: Whether orders cross the spread. Taker is the honest default
            for a signal-driven strategy: if you insist on maker fills you must
            also model non-fills, which this broker does not do.
    """

    starting_cash: float
    venue: VenueCosts
    is_taker: bool = True

    cash: float = field(init=False)
    position: float = field(init=False, default=0.0)
    fills: list[Fill] = field(default_factory=list, init=False)
    rejections: list[Rejection] = field(default_factory=list, init=False)
    total_fees: float = field(init=False, default=0.0)
    total_slippage: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        self.cash = float(self.starting_cash)

    # -- state -------------------------------------------------------------

    def equity(self, mark_price: float) -> float:
        """Total account value marked at the given price."""
        return self.cash + self.position * mark_price

    def position_value(self, mark_price: float) -> float:
        return self.position * mark_price

    # -- execution ---------------------------------------------------------

    def _reject(
        self,
        timestamp: datetime,
        side: Side,
        quantity: float,
        price: float,
        reason: RejectReason,
        detail: str,
    ) -> None:
        self.rejections.append(Rejection(timestamp, side, quantity, price, reason, detail))
        log.debug("rejected %s %.8f @ %.2f: %s", side.value, quantity, price, detail)

    def execute(
        self,
        timestamp: datetime,
        side: Side,
        quantity: float,
        reference_price: float,
    ) -> Fill | None:
        """Attempt to execute an order. Returns the Fill, or None if rejected.

        `reference_price` is the fair mid at execution time (in a backtest, the
        next bar's open). Slippage is applied against it in the direction that
        hurts: buys fill higher, sells fill lower.
        """
        if quantity <= 0:
            self._reject(timestamp, side, quantity, reference_price, RejectReason.NON_POSITIVE_SIZE,
                         f"quantity {quantity} is not positive")
            return None

        notional = quantity * reference_price

        if notional < self.venue.min_notional_usd:
            self._reject(
                timestamp, side, quantity, reference_price, RejectReason.BELOW_MIN_NOTIONAL,
                f"notional ${notional:.2f} < venue minimum ${self.venue.min_notional_usd:.2f}",
            )
            return None

        slip_bps = self.venue.slippage_bps(notional, is_taker=self.is_taker)
        direction = 1.0 if side is Side.BUY else -1.0
        fill_price = reference_price * (1.0 + direction * slip_bps * BPS)
        fill_notional = quantity * fill_price
        fee = fill_notional * self.venue.fee_bps(is_taker=self.is_taker) * BPS
        slippage_cost = abs(fill_notional - notional)

        if side is Side.BUY:
            total_debit = fill_notional + fee
            if total_debit > self.cash + 1e-9:
                self._reject(
                    timestamp, side, quantity, reference_price, RejectReason.INSUFFICIENT_CASH,
                    f"need ${total_debit:.2f}, have ${self.cash:.2f}",
                )
                return None
            self.cash -= total_debit
            self.position += quantity
        else:
            if quantity > self.position + 1e-12:
                self._reject(
                    timestamp, side, quantity, reference_price, RejectReason.INSUFFICIENT_POSITION,
                    f"sell {quantity:.8f} exceeds position {self.position:.8f}",
                )
                return None
            self.cash += fill_notional - fee
            self.position -= quantity
            if abs(self.position) < 1e-12:
                self.position = 0.0

        self.total_fees += fee
        self.total_slippage += slippage_cost

        fill = Fill(
            timestamp=timestamp,
            side=side,
            quantity=quantity,
            reference_price=reference_price,
            fill_price=fill_price,
            notional=fill_notional,
            fee=fee,
            slippage_cost=slippage_cost,
            cash_after=self.cash,
            position_after=self.position,
        )
        self.fills.append(fill)
        return fill

    # -- convenience -------------------------------------------------------

    def target_position_value(
        self, timestamp: datetime, target_value: float, price: float
    ) -> Fill | None:
        """Rebalance toward holding `target_value` USD of the asset.

        Returns the resulting Fill, or None if no trade was needed or the order
        was rejected. Rejections are recorded on the broker either way.
        """
        current_value = self.position_value(price)
        delta_value = target_value - current_value

        # Skip trades too small to clear the venue minimum. Checking here as
        # well as in execute() keeps the rejection log meaningful rather than
        # flooded with sub-cent noise.
        if abs(delta_value) < max(self.venue.min_notional_usd, 1e-8):
            return None

        quantity = abs(delta_value) / price
        side = Side.BUY if delta_value > 0 else Side.SELL

        if side is Side.BUY:
            # Leave room for the fee so a full-size buy is not rejected for
            # being a few cents short.
            fee_rate = self.venue.fee_bps(is_taker=self.is_taker) * BPS
            slip_rate = self.venue.slippage_bps(abs(delta_value), is_taker=self.is_taker) * BPS
            affordable = self.cash / (price * (1.0 + slip_rate) * (1.0 + fee_rate))
            quantity = min(quantity, affordable)
        else:
            quantity = min(quantity, self.position)

        return self.execute(timestamp, side, quantity, price)

    def liquidate(self, timestamp: datetime, price: float) -> Fill | None:
        """Close the position entirely. Used to flatten at end of backtest."""
        if self.position <= 0:
            return None
        return self.execute(timestamp, Side.SELL, self.position, price)

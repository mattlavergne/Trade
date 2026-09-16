"""Multi-asset universe construction.

Diversification is the only thing in finance that reliably improves risk-adjusted
returns without requiring you to predict anything. A trend strategy on one asset
is a coin flip with a story attached; the same strategy across 15 weakly
correlated assets is the managed-futures industry.

This module builds an aligned price panel across many symbols so strategies can
be evaluated on a portfolio rather than a lucky single market.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .data import fetch_ohlcv

log = logging.getLogger(__name__)

# Liquid USD pairs with multi-year daily history on Coinbase. Chosen for data
# depth and liquidity, NOT for backtest performance -- picking the universe
# after seeing results is selection bias, and it is how people fool themselves.
DEFAULT_UNIVERSE = [
    "BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "ADA/USD", "AVAX/USD",
    "LINK/USD", "LTC/USD", "DOT/USD", "ATOM/USD", "UNI/USD", "AAVE/USD",
    "ALGO/USD", "XLM/USD",
]


class Panel:
    """Aligned OHLCV data across multiple symbols.

    Stored as a dict of field -> DataFrame(date x symbol) so a strategy can
    operate on whole cross-sections at once.
    """

    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        if not frames:
            raise ValueError("Panel needs at least one symbol")
        self.symbols = sorted(frames)
        index = None
        for frame in frames.values():
            index = frame.index if index is None else index.union(frame.index)

        self.open = pd.DataFrame(index=index, columns=self.symbols, dtype=float)
        self.high = pd.DataFrame(index=index, columns=self.symbols, dtype=float)
        self.low = pd.DataFrame(index=index, columns=self.symbols, dtype=float)
        self.close = pd.DataFrame(index=index, columns=self.symbols, dtype=float)
        self.volume = pd.DataFrame(index=index, columns=self.symbols, dtype=float)

        for symbol, frame in frames.items():
            for field in ("open", "high", "low", "close", "volume"):
                getattr(self, field)[symbol] = frame[field].reindex(index)

        # Forward-fill gaps (exchange outages, thin days) but never back-fill:
        # back-filling leaks future information into the past.
        for field in ("open", "high", "low", "close", "volume"):
            setattr(self, field, getattr(self, field).ffill())

    def __len__(self) -> int:
        return len(self.close)

    @property
    def index(self) -> pd.Index:
        return self.close.index

    def returns(self) -> pd.DataFrame:
        return self.close.pct_change()

    def tradeable(self) -> pd.DataFrame:
        """Boolean mask: True where an asset has enough history to trade.

        An asset that has not listed yet must be untradeable, or the backtest
        will happily buy something that did not exist.
        """
        return self.close.notna() & self.close.shift(1).notna()

    def slice(self, end: int) -> "Panel":
        """Return a Panel truncated to the first `end` bars (for walk-forward)."""
        frames = {
            symbol: pd.DataFrame({
                "open": self.open[symbol].iloc[:end],
                "high": self.high[symbol].iloc[:end],
                "low": self.low[symbol].iloc[:end],
                "close": self.close[symbol].iloc[:end],
                "volume": self.volume[symbol].iloc[:end],
            })
            for symbol in self.symbols
        }
        return Panel(frames)

    def correlation_summary(self) -> pd.DataFrame:
        return self.returns().corr()

    def describe(self) -> str:
        corr = self.correlation_summary()
        upper = corr.to_numpy()[np.triu_indices(len(self.symbols), k=1)]
        return (
            f"{len(self.symbols)} symbols, {len(self)} bars "
            f"({self.index[0].date()} to {self.index[-1].date()}), "
            f"mean pairwise correlation {np.nanmean(upper):.2f}"
        )


def load_universe(
    symbols: list[str] | None = None,
    *,
    venue: str = "coinbase",
    timeframe: str = "1d",
    days: int = 1500,
    use_cache: bool = True,
    min_bars: int = 400,
) -> Panel:
    """Fetch and align bars for every symbol that has enough history."""
    symbols = symbols or DEFAULT_UNIVERSE
    frames: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        try:
            frame = fetch_ohlcv(venue, symbol, timeframe, days=days, use_cache=use_cache)
            if len(frame) < min_bars:
                log.warning("skipping %s: only %d bars (< %d)", symbol, len(frame), min_bars)
                continue
            frames[symbol] = frame
        except Exception as exc:  # noqa: BLE001 - a dead symbol must not kill the load
            log.warning("skipping %s: %s", symbol, exc)

    if not frames:
        raise RuntimeError("no symbols loaded; check connectivity and symbol names")

    panel = Panel(frames)
    log.info("universe: %s", panel.describe())
    return panel

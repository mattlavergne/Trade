"""Historical and live market data via ccxt, with an on-disk cache.

Exchanges rate-limit aggressively and paginate OHLCV in chunks. This module
handles both, and caches to CSV so repeated backtests do not re-hammer the API.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import ccxt
import pandas as pd

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache"

TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def make_exchange(venue: str) -> ccxt.Exchange:
    """Build a ccxt client, honouring an HTTPS proxy if one is configured.

    ccxt does not read HTTPS_PROXY from the environment on its own, so we wire
    it explicitly. Harmless when no proxy is set.
    """
    if not hasattr(ccxt, venue):
        raise ValueError(f"ccxt has no exchange named {venue!r}")
    exchange = getattr(ccxt, venue)({"enableRateLimit": True, "timeout": 30_000})
    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if proxy:
        exchange.httpsProxy = proxy
    return exchange


def _cache_path(venue: str, symbol: str, timeframe: str) -> Path:
    safe_symbol = symbol.replace("/", "-")
    return CACHE_DIR / f"{venue}_{safe_symbol}_{timeframe}.csv"


def fetch_ohlcv(
    venue: str,
    symbol: str,
    timeframe: str = "1h",
    *,
    days: int = 365,
    use_cache: bool = True,
    max_pages: int = 500,
) -> pd.DataFrame:
    """Fetch OHLCV bars, paginating backwards from now.

    Returns a DataFrame indexed by UTC timestamp with open/high/low/close/volume
    columns, sorted ascending and de-duplicated.
    """
    if timeframe not in TIMEFRAME_MS:
        raise ValueError(f"Unsupported timeframe {timeframe!r}. Use one of {sorted(TIMEFRAME_MS)}")

    path = _cache_path(venue, symbol, timeframe)
    if use_cache and path.exists():
        cached = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
        span_days = (cached.index[-1] - cached.index[0]).total_seconds() / 86_400
        if span_days >= days * 0.95:
            log.info("cache hit %s (%d bars, %.0f days)", path.name, len(cached), span_days)
            return cached.loc[cached.index >= cached.index[-1] - pd.Timedelta(days=days)]

    exchange = make_exchange(venue)
    step_ms = TIMEFRAME_MS[timeframe]
    since = exchange.milliseconds() - days * 86_400_000
    rows: list[list[float]] = []
    cursor = since

    for _ in range(max_pages):
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=1000)
        if not batch:
            break
        rows.extend(batch)
        next_cursor = batch[-1][0] + step_ms
        if next_cursor <= cursor or batch[-1][0] >= exchange.milliseconds() - step_ms:
            break
        cursor = next_cursor
        time.sleep(exchange.rateLimit / 1000.0)

    if not rows:
        raise RuntimeError(f"No OHLCV returned for {symbol} {timeframe} on {venue}")

    frame = pd.DataFrame(rows, columns=OHLCV_COLUMNS)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path)
    log.info("fetched %d bars for %s %s on %s", len(frame), symbol, timeframe, venue)
    return frame


def fetch_tickers(venues: list[str], symbol: str = "BTC/USD") -> dict[str, float]:
    """Current mid price for one symbol across several venues.

    Used by the cross-venue spread check, which exists to demonstrate how small
    real dislocations are relative to fees.
    """
    out: dict[str, float] = {}
    for venue in venues:
        try:
            ticker = make_exchange(venue).fetch_ticker(symbol)
            price = ticker.get("last") or ticker.get("close")
            if price:
                out[venue] = float(price)
        except Exception as exc:  # noqa: BLE001 - one bad venue must not kill the sweep
            log.warning("ticker fetch failed for %s: %s", venue, exc)
    return out

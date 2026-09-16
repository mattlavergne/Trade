"""Live paper trading: real market prices, simulated money.

This runs the same strategy code and the same broker as the backtester, but
against live prices, and it persists state to disk so it survives restarts.

Why this step matters and cannot be skipped: a backtest is a story you tell
yourself about the past, fitted (consciously or not) to data you have already
seen. Live paper trading is the first time a strategy meets data that did not
exist when it was written. Most strategies that look good in backtest fail
here. That failure costs nothing, which is the entire point.

NOTHING IN THIS MODULE PLACES A REAL ORDER. There is no API key, no signing,
no private endpoint. It physically cannot spend money.
"""

from __future__ import annotations

import json
import logging
import signal
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .broker import PaperBroker, Side
from .costs import VenueCosts, get_venue
from .data import fetch_ohlcv, make_exchange
from .strategy import Strategy

log = logging.getLogger(__name__)

STATE_DIR = Path(__file__).resolve().parent.parent / "state"


@dataclass
class LiveState:
    """Everything needed to resume a paper session after a restart."""

    session_id: str
    strategy: str
    symbol: str
    timeframe: str
    venue: str
    starting_cash: float
    cash: float
    position: float
    total_fees: float
    total_slippage: float
    bars_seen: int
    last_bar_ts: str | None
    started_at: str
    updated_at: str
    equity_log: list[dict] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)


class LivePaperTrader:
    """Poll for new bars, run the strategy, simulate fills."""

    def __init__(
        self,
        strategy: Strategy,
        *,
        symbol: str = "BTC/USD",
        timeframe: str = "1h",
        data_venue: str = "binanceus",
        cost_venue: str = "kraken",
        starting_cash: float = 50.0,
        session_id: str | None = None,
        history_days: int = 30,
    ) -> None:
        self.strategy = strategy
        self.symbol = symbol
        self.timeframe = timeframe
        self.data_venue = data_venue
        self.venue: VenueCosts = get_venue(cost_venue)
        self.history_days = history_days
        self.session_id = session_id or f"{strategy.name}_{symbol.replace('/', '-')}_{timeframe}"
        self.broker = PaperBroker(starting_cash=starting_cash, venue=self.venue)
        self.state_path = STATE_DIR / f"{self.session_id}.json"
        self.equity_log: list[dict] = []
        self.trade_log: list[dict] = []
        self.last_bar_ts: pd.Timestamp | None = None
        self.bars_seen = 0
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._stop = False
        self._exchange = make_exchange(data_venue)

    # -- persistence -------------------------------------------------------

    def load(self) -> bool:
        """Restore a previous session. Returns True if state was found."""
        if not self.state_path.exists():
            return False
        raw = json.loads(self.state_path.read_text())
        self.broker.cash = raw["cash"]
        self.broker.position = raw["position"]
        self.broker.total_fees = raw["total_fees"]
        self.broker.total_slippage = raw["total_slippage"]
        self.bars_seen = raw["bars_seen"]
        self.started_at = raw["started_at"]
        self.equity_log = raw.get("equity_log", [])
        self.trade_log = raw.get("trade_log", [])
        if raw.get("last_bar_ts"):
            self.last_bar_ts = pd.Timestamp(raw["last_bar_ts"])
        log.info("resumed session %s: %d bars seen, equity components restored",
                 self.session_id, self.bars_seen)
        return True

    def save(self, mark_price: float) -> None:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        state = LiveState(
            session_id=self.session_id,
            strategy=self.strategy.describe(),
            symbol=self.symbol,
            timeframe=self.timeframe,
            venue=self.venue.name,
            starting_cash=self.broker.starting_cash,
            cash=self.broker.cash,
            position=self.broker.position,
            total_fees=self.broker.total_fees,
            total_slippage=self.broker.total_slippage,
            bars_seen=self.bars_seen,
            last_bar_ts=self.last_bar_ts.isoformat() if self.last_bar_ts is not None else None,
            started_at=self.started_at,
            updated_at=datetime.now(timezone.utc).isoformat(),
            equity_log=self.equity_log[-5000:],
            trade_log=self.trade_log[-2000:],
        )
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(state), indent=2))
        tmp.replace(self.state_path)  # atomic: a crash mid-write cannot corrupt state

    # -- trading loop ------------------------------------------------------

    def _fetch_recent(self) -> pd.DataFrame:
        return fetch_ohlcv(
            self.data_venue, self.symbol, self.timeframe,
            days=self.history_days, use_cache=False,
        )

    def step(self) -> dict | None:
        """Process one poll. Returns a summary dict if a new bar was handled."""
        bars = self._fetch_recent()
        if len(bars) < 2:
            return None

        # The final bar is still forming; act only on completed bars.
        completed = bars.iloc[:-1]
        latest_ts = completed.index[-1]
        if self.last_bar_ts is not None and latest_ts <= self.last_bar_ts:
            return None  # no new completed bar yet

        exposure = self.strategy.target_exposure(completed)
        target_fraction = float(exposure.iloc[-1])
        price = float(bars["open"].iloc[-1])  # execute at the forming bar's open
        now = datetime.now(timezone.utc)

        before_position = self.broker.position
        desired_value = self.broker.equity(price) * target_fraction
        fill = self.broker.target_position_value(now, desired_value, price)

        self.last_bar_ts = latest_ts
        self.bars_seen += 1
        mark = float(bars["close"].iloc[-1])
        equity = self.broker.equity(mark)

        self.equity_log.append({
            "ts": now.isoformat(), "bar_ts": latest_ts.isoformat(),
            "price": mark, "equity": equity, "position": self.broker.position,
            "cash": self.broker.cash, "target_exposure": target_fraction,
        })

        if fill is not None:
            self.trade_log.append({
                "ts": now.isoformat(), "side": fill.side.value,
                "quantity": fill.quantity, "fill_price": fill.fill_price,
                "notional": fill.notional, "fee": fill.fee,
                "slippage": fill.slippage_cost, "equity_after": equity,
            })
            log.info("FILL %s %.8f @ %.2f  fee $%.4f  equity $%.2f",
                     fill.side.value.upper(), fill.quantity, fill.fill_price, fill.fee, equity)
        elif abs(target_fraction - (1.0 if before_position > 0 else 0.0)) > 0.5:
            reason = self.broker.rejections[-1].reason.value if self.broker.rejections else "n/a"
            log.info("no fill (target %.2f) - last rejection: %s", target_fraction, reason)

        self.save(mark)
        return {
            "bar_ts": latest_ts, "price": mark, "equity": equity,
            "position": self.broker.position, "target": target_fraction,
            "filled": fill is not None,
        }

    def run(self, *, poll_seconds: int = 60, max_polls: int | None = None) -> None:
        """Poll until interrupted. Ctrl-C shuts down cleanly and saves state."""

        def handle_stop(signum, frame):  # noqa: ARG001
            log.info("shutdown requested; saving state")
            self._stop = True

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

        log.info("live paper trading %s | %s %s | costs: %s | starting $%.2f",
                 self.strategy.describe(), self.symbol, self.timeframe,
                 self.venue.name, self.broker.starting_cash)
        log.info("NO REAL ORDERS ARE PLACED. This process has no exchange credentials.")

        polls = 0
        while not self._stop and (max_polls is None or polls < max_polls):
            try:
                result = self.step()
                if result:
                    log.info("bar %s | price $%.2f | equity $%.2f | pos %.8f | target %.0f%%",
                             result["bar_ts"], result["price"], result["equity"],
                             result["position"], result["target"] * 100)
            except Exception as exc:  # noqa: BLE001 - a live loop must not die on a blip
                log.warning("poll failed (%s); continuing", exc)
            polls += 1
            if self._stop or (max_polls is not None and polls >= max_polls):
                break
            time.sleep(poll_seconds)

        try:
            ticker = self._exchange.fetch_ticker(self.symbol)
            self.save(float(ticker.get("last") or ticker.get("close") or 0.0))
        except Exception:  # noqa: BLE001
            pass
        log.info("stopped. state at %s", self.state_path)

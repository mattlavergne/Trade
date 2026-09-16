#!/usr/bin/env python3
"""Command line interface for the trading research harness.

    python cli.py venues                 # cost table: the hurdle rate per venue
    python cli.py spread-check           # live cross-venue spread vs fees
    python cli.py backtest --strategy sma_cross --venue kraken
    python cli.py compare --venue kraken # every strategy, side by side
    python cli.py paper --strategy donchian --poll 60
    python cli.py status                 # live paper session state
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import trader.strategies  # noqa: F401 - populates the strategy registry
from trader.costs import VENUES, breakeven_move_pct, get_venue
from trader.data import fetch_ohlcv, fetch_tickers
from trader.engine import run_backtest
from trader.live import STATE_DIR, LivePaperTrader
from trader.report import print_result, write_html_report
from trader.strategy import available_strategies, get_strategy


def _parse_params(pairs: list[str] | None) -> dict:
    """Parse --param fast=10 --param slow=40 into typed kwargs."""
    out: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"bad --param {pair!r}; expected name=value")
        key, value = pair.split("=", 1)
        try:
            out[key] = int(value)
        except ValueError:
            try:
                out[key] = float(value)
            except ValueError:
                out[key] = value
    return out


def cmd_venues(args: argparse.Namespace) -> int:
    print()
    print(f"{'venue':18s}{'maker':>9s}{'taker':>9s}{'min order':>11s}{'round trip':>12s}{'breakeven':>11s}")
    print("-" * 70)
    for name, venue in VENUES.items():
        print(f"{name:18s}{venue.maker_bps:8.1f}b{venue.taker_bps:8.1f}b"
              f"{venue.min_notional_usd:10.2f}${venue.round_trip_bps():10.1f}b"
              f"{breakeven_move_pct(venue):10.2f}%")
    print()
    print("  'breakeven' is how far price must move in your favour for one round")
    print("  trip to make zero profit. If your signal predicts smaller moves than")
    print("  this, the strategy loses money by construction.")
    print()
    print("  Risk-budget check: sound risk management risks 1-2% per trade.")
    print("  On a $50 account that is $0.50-$1.00. Compare that to the fee on a")
    print("  $50 round trip below -- if the fee is comparable, the strategy")
    print("  cannot be run responsibly at this size, on any market.")
    print()
    for name, venue in VENUES.items():
        fee = 50.0 * venue.round_trip_bps() * 1e-4
        verdict = "UNRUNNABLE" if fee >= 0.50 else "viable"
        print(f"    {name:18s} $50 round trip costs ${fee:5.2f}   {verdict}")
    print()
    return 0


def cmd_spread_check(args: argparse.Namespace) -> int:
    """Measure the real cross-venue spread and compare it to the fee floor."""
    venues = args.venues or ["kraken", "coinbase", "binanceus"]
    print(f"\nFetching live {args.symbol} prices from {', '.join(venues)}...\n")
    prices = fetch_tickers(venues, args.symbol)
    if len(prices) < 2:
        print("Need at least two venues to compare. Check connectivity.")
        return 1

    for venue, price in sorted(prices.items(), key=lambda kv: kv[1]):
        print(f"  {venue:12s} ${price:,.2f}")

    cheapest = min(prices, key=prices.get)
    dearest = max(prices, key=prices.get)
    spread = prices[dearest] - prices[cheapest]
    spread_bps = spread / prices[cheapest] * 1e4

    buy_fee = get_venue(cheapest).taker_bps if cheapest in VENUES else 40.0
    sell_fee = get_venue(dearest).taker_bps if dearest in VENUES else 40.0
    fee_bps = buy_fee + sell_fee

    print()
    print(f"  Widest gap:      buy {cheapest} / sell {dearest}")
    print(f"  Gross spread:    ${spread:,.2f}  ({spread_bps:.2f} bps)")
    print(f"  Taker fees:      {fee_bps:.1f} bps  ({buy_fee:.0f} buy + {sell_fee:.0f} sell)")
    print(f"  Net edge:        {spread_bps - fee_bps:+.2f} bps")
    print()
    if spread_bps > fee_bps:
        print("  Gross gap exceeds fees this instant -- but you would still need to")
        print("  hold inventory on both venues, execute both legs before the gap")
        print("  closes (milliseconds), and beat firms with colocated hardware.")
    else:
        print(f"  The gap is {fee_bps / max(spread_bps, 0.01):.0f}x SMALLER than the fees required")
        print("  to capture it. Every such trade is a guaranteed loss. This is the")
        print("  normal state of affairs, and it is why the 'scan 50 markets for")
        print("  mispricings' story does not survive contact with a fee schedule.")
    print()
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    strategy = get_strategy(args.strategy, **_parse_params(args.param))
    bars = fetch_ohlcv(args.data_venue, args.symbol, args.timeframe,
                       days=args.days, use_cache=not args.no_cache)
    result = run_backtest(bars, strategy, get_venue(args.venue),
                          starting_cash=args.cash, timeframe=args.timeframe,
                          symbol=args.symbol)
    print_result(result)
    if args.html:
        path = write_html_report(result)
        print(f"  HTML report: {path}\n")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    bars = fetch_ohlcv(args.data_venue, args.symbol, args.timeframe,
                       days=args.days, use_cache=not args.no_cache)
    venue = get_venue(args.venue)
    print(f"\n  {args.symbol} {args.timeframe} | {len(bars)} bars | "
          f"{bars.index[0].date()} to {bars.index[-1].date()} | venue {args.venue} "
          f"| ${args.cash:.2f} start\n")
    header = (f"  {'strategy':18s}{'net ret':>10s}{'vs B&H':>10s}{'maxDD':>9s}"
              f"{'Sharpe':>8s}{'trips':>7s}{'fees':>9s}{'end $':>9s}")
    print(header)
    print("  " + "-" * (len(header) - 2))
    rows = []
    for name in available_strategies():
        result = run_backtest(bars, get_strategy(name), venue,
                              starting_cash=args.cash, timeframe=args.timeframe,
                              symbol=args.symbol)
        m = result.metrics
        rows.append((name, result))
        print(f"  {name:18s}{m.total_return_pct:9.2f}%{result.excess_return_pct:9.2f}%"
              f"{m.max_drawdown_pct:8.2f}%{m.sharpe:8.2f}{m.num_round_trips:7d}"
              f"{m.total_fees:8.2f}${m.ending_equity:8.2f}")
    winners = [r for _, r in rows if r.beat_benchmark]
    print()
    if winners:
        print(f"  {len(winners)} of {len(rows)} strategies beat buy-and-hold net of costs.")
        print("  Beating it on ONE historical sample is weak evidence. Re-run over")
        print("  different date ranges before believing it.")
    else:
        print("  NONE of these strategies beat buy-and-hold net of costs.")
        print("  That is the typical result, and it is the honest one.")
    print()
    return 0


def cmd_paper(args: argparse.Namespace) -> int:
    strategy = get_strategy(args.strategy, **_parse_params(args.param))
    trader_ = LivePaperTrader(
        strategy, symbol=args.symbol, timeframe=args.timeframe,
        data_venue=args.data_venue, cost_venue=args.venue,
        starting_cash=args.cash, session_id=args.session,
    )
    if not args.fresh:
        trader_.load()
    trader_.run(poll_seconds=args.poll, max_polls=args.max_polls)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    if not STATE_DIR.exists() or not any(STATE_DIR.glob("*.json")):
        print("\n  No live paper sessions found. Start one with:")
        print("    python cli.py paper --strategy donchian\n")
        return 0
    for path in sorted(STATE_DIR.glob("*.json")):
        state = json.loads(path.read_text())
        start = state["starting_cash"]
        last_price = state["equity_log"][-1]["price"] if state["equity_log"] else 0.0
        equity = state["cash"] + state["position"] * last_price
        pnl_pct = (equity / start - 1.0) * 100.0 if start else 0.0
        print(f"\n  session        {state['session_id']}")
        print(f"  strategy       {state['strategy']}")
        print(f"  market         {state['symbol']} {state['timeframe']} (costs: {state['venue']})")
        print(f"  started        {state['started_at'][:19]}")
        print(f"  updated        {state['updated_at'][:19]}")
        print(f"  bars seen      {state['bars_seen']}")
        print(f"  trades         {len(state['trade_log'])}")
        print(f"  cash           ${state['cash']:,.2f}")
        print(f"  position       {state['position']:.8f} @ ${last_price:,.2f}")
        print(f"  equity         ${equity:,.2f}  ({pnl_pct:+.2f}% from ${start:,.2f})")
        print(f"  costs paid     ${state['total_fees'] + state['total_slippage']:,.4f}")
        if len(state["trade_log"]) < 30:
            print(f"  NOTE: {len(state['trade_log'])} trades is far too few to judge anything.")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest and paper-trade research harness. Places no real orders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_market_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--symbol", default="BTC/USD")
        p.add_argument("--timeframe", default="1h", choices=["1m", "5m", "15m", "1h", "4h", "1d"])
        p.add_argument("--data-venue", default="binanceus",
                       help="where bars come from (binanceus and coinbase paginate deepest)")
        p.add_argument("--venue", default="kraken", choices=sorted(VENUES),
                       help="which fee schedule to charge against fills")
        p.add_argument("--cash", type=float, default=50.0)
        p.add_argument("--days", type=int, default=365)
        p.add_argument("--no-cache", action="store_true")

    p = sub.add_parser("venues", help="show fee schedules and breakeven hurdles")
    p.set_defaults(func=cmd_venues)

    p = sub.add_parser("spread-check", help="live cross-venue spread vs fee floor")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--venues", nargs="*")
    p.set_defaults(func=cmd_spread_check)

    p = sub.add_parser("backtest", help="run one strategy over history")
    add_market_args(p)
    p.add_argument("--strategy", required=True, choices=available_strategies())
    p.add_argument("--param", action="append", help="strategy param, e.g. --param fast=10")
    p.add_argument("--html", action="store_true", help="also write an HTML report")
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser("compare", help="run every strategy side by side")
    add_market_args(p)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("paper", help="live paper trade (simulated money, real prices)")
    add_market_args(p)
    p.add_argument("--strategy", required=True, choices=available_strategies())
    p.add_argument("--param", action="append")
    p.add_argument("--poll", type=int, default=60, help="seconds between polls")
    p.add_argument("--max-polls", type=int, default=None)
    p.add_argument("--session", default=None)
    p.add_argument("--fresh", action="store_true", help="ignore saved state")
    p.set_defaults(func=cmd_paper)

    p = sub.add_parser("status", help="show live paper session state")
    p.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

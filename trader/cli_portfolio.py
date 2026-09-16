"""Portfolio-level CLI commands: multi-asset backtests, walk-forward, funding."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .costs import get_venue
from .funding import (
    FUNDING_PERIODS_PER_YEAR,
    breakeven_holding_periods,
    carry_trade_pnl,
    fetch_funding_history,
    summarise_funding,
)
from .portfolio import run_portfolio_backtest
from .risk import RiskConfig
from .strategies.portfolio_strategies import PORTFOLIO_STRATEGIES
from .universe import DEFAULT_UNIVERSE, load_universe
from .validation import (
    block_bootstrap_pvalue,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    walk_forward,
)


def _curve_stats(curve: pd.Series, returns: pd.Series) -> dict:
    years = len(curve) / 365
    cagr = (curve.iloc[-1] / curve.iloc[0]) ** (1 / years) - 1 if years > 0 else 0.0
    drawdown = float((curve / curve.cummax() - 1.0).min())
    return {
        "total": (curve.iloc[-1] / curve.iloc[0] - 1.0) * 100.0,
        "cagr": cagr * 100.0,
        "vol": float(returns.std() * np.sqrt(365)) * 100.0,
        "sharpe": sharpe_ratio(returns),
        "dd": drawdown * 100.0,
        "calmar": cagr / abs(drawdown) if drawdown < 0 else 0.0,
        "psr": probabilistic_sharpe_ratio(returns),
    }


def _risk_config(args) -> RiskConfig:
    return RiskConfig(
        target_volatility=args.target_vol,
        max_leverage=args.max_leverage,
        no_trade_band=args.band,
    )


def cmd_portfolio(args) -> int:
    panel = load_universe(days=args.days, use_cache=not args.no_cache)
    venue = get_venue(args.venue)
    config = _risk_config(args)

    print(f"\n  {panel.describe()}")
    print(f"  venue {args.venue} | target vol {config.target_volatility:.0%} | "
          f"band {config.no_trade_band:.2f} | ${args.cash:,.0f} start\n")
    header = (f"  {'strategy':16s}{'total':>9s}{'CAGR':>8s}{'vol':>7s}{'Sharpe':>8s}"
              f"{'maxDD':>8s}{'Calmar':>8s}{'PSR':>7s}{'turn/y':>8s}{'cost%':>7s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    results = {}
    for name, cls in PORTFOLIO_STRATEGIES.items():
        result = run_portfolio_backtest(panel, cls(), venue, config=config,
                                        starting_cash=args.cash)
        returns = result.daily_returns()
        results[name] = result
        s = _curve_stats(result.equity_curve, returns)
        years = len(result.equity_curve) / 365
        print(f"  {name:16s}{s['total']:8.1f}%{s['cagr']:7.1f}%{s['vol']:6.1f}%"
              f"{s['sharpe']:8.2f}{s['dd']:7.1f}%{s['calmar']:8.2f}{s['psr']:6.1%}"
              f"{result.turnover.sum() / years:8.1f}"
              f"{result.total_cost / args.cash * 100:6.1f}%")

    reference = next(iter(results.values()))
    bench_returns = reference.benchmark_curve.pct_change().dropna()
    s = _curve_stats(reference.benchmark_curve, bench_returns)
    print("  " + "-" * (len(header) - 2))
    print(f"  {'RAW BUY & HOLD':16s}{s['total']:8.1f}%{s['cagr']:7.1f}%{s['vol']:6.1f}%"
          f"{s['sharpe']:8.2f}{s['dd']:7.1f}%{s['calmar']:8.2f}{s['psr']:6.1%}"
          f"{0.0:8.1f}{0.0:6.1f}%")

    print("\n  'hold' is the SAME buy-and-hold portfolio run through the volatility")
    print("  targeting engine. Compare those two rows: that difference is what risk")
    print("  management contributes, with no forecasting whatsoever.")
    print("\n  Any strategy row beating 'hold' is doing so having been chosen after")
    print("  the fact. Run 'walkforward' before believing it.\n")
    return 0


def cmd_walkforward(args) -> int:
    panel = load_universe(days=args.days, use_cache=not args.no_cache)
    venue = get_venue(args.venue)
    config = _risk_config(args)

    factory = PORTFOLIO_STRATEGIES[args.strategy]
    if args.strategy == "trend_filtered":
        grid = [{"lookback": lb, "top_n": n, "trend_window": tw}
                for lb in (30, 60, 120) for n in (3, 5) for tw in (100, 200)]
    elif args.strategy == "xsmom":
        grid = [{"lookback": lb, "top_n": n}
                for lb in (30, 60, 120) for n in (3, 5)]
    elif args.strategy == "tsmom":
        grid = [{"lookback": lb, "vol_scale": vs}
                for lb in (30, 60, 90, 120, 180) for vs in (True, False)]
    else:
        grid = [{}]

    # Duplicate parameter sets would inflate the trial count used by the
    # Deflated Sharpe and waste half the search. Guard against it.
    unique = []
    for params in grid:
        if params not in unique:
            unique.append(params)
    grid = unique

    print(f"\n  {panel.describe()}")
    print(f"  walk-forward: train {args.train} bars, test {args.test} bars, "
          f"{len(grid)} parameter combinations\n")

    result = walk_forward(panel, factory, grid, venue, config=config,
                          train_bars=args.train, test_bars=args.test,
                          starting_cash=args.cash)
    if not result.windows:
        print("  Not enough history for these window sizes.\n")
        return 1

    print(f"  {'test period':>24s}{'IS Sharpe':>11s}{'OOS Sharpe':>12s}{'OOS ret':>10s}  params")
    print("  " + "-" * 96)
    for w in result.windows:
        period = f"{w.test_start.date()} to {w.test_end.date()}"
        print(f"  {period:>24s}{w.in_sample_sharpe:11.2f}{w.out_of_sample_sharpe:12.2f}"
              f"{w.out_of_sample_return_pct:9.1f}%  {w.best_params}")

    combined = result.combined_returns
    print(f"\n  in-sample mean Sharpe    {result.in_sample_mean_sharpe:6.2f}")
    print(f"  OUT-OF-SAMPLE Sharpe     {result.out_of_sample_sharpe:6.2f}")
    print(f"  degradation              {result.degradation:6.2f}")
    print(f"  windows profitable       {result.consistency:6.0%}")
    print(f"  bootstrap p-value        {block_bootstrap_pvalue(combined, num_samples=2000):6.3f}")
    print(f"  deflated Sharpe          {deflated_sharpe_ratio(combined, result.num_trials):6.1%}"
          f"  ({result.num_trials} configurations tested)")

    issues = result.verdict()
    print("\n  VERDICT:")
    if not issues:
        print("    No red flags. That is necessary, not sufficient -- paper trade it next.")
    for issue in issues:
        print(f"    - {issue}")
    print()
    return 0


def cmd_funding(args) -> int:
    symbols = args.symbols or ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT",
                               "DOGE/USDT:USDT", "XRP/USDT:USDT"]
    print(f"\n  Perpetual funding on {args.venue}. Positive means longs pay shorts,")
    print("  i.e. a delta-neutral long-spot/short-perp position earns it.\n")
    print(f"  {'symbol':18s}{'periods':>9s}{'span':>7s}{'ann.%':>8s}{'%pos':>7s}"
          f"{'bps/8h':>9s}{'worst':>9s}")
    print("  " + "-" * 68)

    collected = {}
    for symbol in symbols:
        try:
            rates = fetch_funding_history(symbol, venue=args.venue, days=args.days)
            summary = summarise_funding(rates)
            if not summary:
                print(f"  {symbol:18s} no data")
                continue
            collected[symbol] = rates
            print(f"  {symbol:18s}{summary['periods']:9d}{summary['span_days']:6d}d"
                  f"{summary['annualised_pct']:7.2f}%{summary['pct_positive']:6.1f}%"
                  f"{summary['mean_bps']:9.3f}{summary['worst_period_bps']:9.2f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {symbol:18s} FAIL {exc}")

    if not collected:
        return 1

    print("\n  CARRY TRADE ECONOMICS (long spot + short perp, delta neutral)")
    print(f"  {'symbol':18s}{'net ann.%':>11s}{'breakeven hold':>16s}{'net Sharpe':>12s}")
    print("  " + "-" * 58)
    for symbol, rates in collected.items():
        pnl = carry_trade_pnl(rates, entry_cost_bps=args.cost_bps,
                              exit_cost_bps=args.cost_bps,
                              capital_efficiency=args.capital_efficiency)
        if pnl.empty:
            continue
        net = pnl["net_return"]
        ann = float(net.mean()) * FUNDING_PERIODS_PER_YEAR * 100.0
        sharpe = (float(net.mean() / net.std()) * np.sqrt(FUNDING_PERIODS_PER_YEAR)
                  if net.std() > 0 else 0.0)
        periods = breakeven_holding_periods(
            float(rates.mean()), entry_cost_bps=args.cost_bps,
            exit_cost_bps=args.cost_bps, capital_efficiency=args.capital_efficiency)
        print(f"  {symbol:18s}{ann:10.2f}%{periods / 3:14.0f}d{sharpe:12.2f}")

    if len(collected) > 1:
        print("\n  Funding correlation across assets (low = diversification works):")
        print(pd.DataFrame(collected).dropna().corr().round(2).to_string(
            float_format=lambda v: f"{v:.2f}").replace("\n", "\n  "))

    print("\n  REALITY CHECK: the Sharpe above still excludes exchange counterparty")
    print("  risk, which is not diversifiable and has destroyed more crypto capital")
    print("  than every bad strategy combined. US persons also cannot legally access")
    print("  perpetual futures on the venues where this trade works.\n")
    return 0


def cmd_universe(args) -> int:
    panel = load_universe(days=args.days, use_cache=not args.no_cache)
    print(f"\n  {panel.describe()}\n")
    corr = panel.correlation_summary()
    print("  Pairwise return correlation:")
    print("  " + corr.round(2).to_string().replace("\n", "\n  "))
    upper = corr.to_numpy()[np.triu_indices(len(panel.symbols), k=1)]
    mean_corr = float(np.nanmean(upper))
    effective = len(panel.symbols) / (1 + (len(panel.symbols) - 1) * mean_corr)
    print(f"\n  Mean pairwise correlation: {mean_corr:.2f}")
    print(f"  Effective independent bets: {effective:.1f} (of {len(panel.symbols)} assets)")
    print("\n  Crypto assets are highly correlated, so holding 14 of them is closer to")
    print(f"  holding {effective:.0f}. Diversification here is far weaker than the asset")
    print("  count suggests -- this is why crypto-only portfolios stay volatile.\n")
    return 0


def register_portfolio_commands(sub, default_cash: float = 10_000.0) -> None:
    """Attach portfolio subcommands to an existing argparse subparser set."""

    def add_common(p):
        p.add_argument("--venue", default="kraken")
        p.add_argument("--days", type=int, default=1500)
        p.add_argument("--cash", type=float, default=default_cash)
        p.add_argument("--target-vol", type=float, default=0.30)
        p.add_argument("--max-leverage", type=float, default=1.0)
        p.add_argument("--band", type=float, default=0.50)
        p.add_argument("--no-cache", action="store_true")

    p = sub.add_parser("portfolio", help="multi-asset backtest, all strategies")
    add_common(p)
    p.set_defaults(func=cmd_portfolio)

    p = sub.add_parser("walkforward", help="out-of-sample validation with overfitting checks")
    add_common(p)
    p.add_argument("--strategy", default="trend_filtered", choices=sorted(PORTFOLIO_STRATEGIES))
    p.add_argument("--train", type=int, default=365)
    p.add_argument("--test", type=int, default=120)
    p.set_defaults(func=cmd_walkforward)

    p = sub.add_parser("funding", help="perpetual funding rates and carry economics")
    p.add_argument("--venue", default="okx")
    p.add_argument("--days", type=int, default=365)
    p.add_argument("--symbols", nargs="*")
    p.add_argument("--cost-bps", type=float, default=20.0,
                   help="one-way cost of establishing both legs, in bps")
    p.add_argument("--capital-efficiency", type=float, default=0.5,
                   help="fraction of capital earning funding after margin buffer")
    p.set_defaults(func=cmd_funding)

    p = sub.add_parser("leverage", help="how much leverage is justified, and its ruin cost")
    add_common(p)
    p.add_argument("--strategy", default="hold", choices=sorted(PORTFOLIO_STRATEGIES))
    p.add_argument("--horizon", type=int, default=365, help="days to project")
    p.add_argument("--paths", type=int, default=10000)
    p.add_argument("--control", action="store_true",
                   help="also show outcomes assuming the edge is zero")
    p.set_defaults(func=cmd_leverage)

    p = sub.add_parser("universe", help="show the asset universe and its correlations")
    p.add_argument("--days", type=int, default=1500)
    p.add_argument("--no-cache", action="store_true")
    p.set_defaults(func=cmd_universe)


def cmd_leverage(args) -> int:
    """How much leverage is justified, and what it costs in ruin probability."""
    from .leverage import analyse, outcome_distribution, zero_edge_control

    panel = load_universe(days=args.days, use_cache=not args.no_cache)
    venue = get_venue(args.venue)
    config = RiskConfig(target_volatility=args.target_vol, no_trade_band=args.band)
    strategy = PORTFOLIO_STRATEGIES[args.strategy]()
    result = run_portfolio_backtest(panel, strategy, venue, config=config,
                                    starting_cash=args.cash)
    returns = result.daily_returns()
    profile = analyse(returns)

    print(f"\n  {strategy.describe()} on {len(panel.symbols)} assets, "
          f"{profile.years:.1f} years\n")
    print(f"  arithmetic return   {profile.arithmetic_return * 100:7.2f}%/yr")
    print(f"  volatility          {profile.volatility * 100:7.2f}%/yr")
    print(f"  volatility drag     {profile.volatility ** 2 / 2 * 100:7.2f}%/yr")
    print(f"  geometric return    {profile.growth(1.0) * 100:7.2f}%/yr   <- what compounds")
    print(f"  Sharpe              {profile.sharpe:7.2f}  +/- {profile.sharpe_stderr:.2f}")
    low, high = profile.sharpe_ci
    print(f"  95% CI for Sharpe   [{low:+.2f}, {high:+.2f}]")

    print(f"\n  Kelly-optimal leverage  {profile.kelly_leverage:.2f}x")
    print(f"  growth at Kelly         {profile.growth_at_kelly * 100:.1f}%/yr")
    print(f"  growth at 2x Kelly      {profile.growth(2 * profile.kelly_leverage) * 100:.1f}%/yr"
          "   <- zero growth, double the risk")
    k_low, k_high = profile.kelly_ci
    print(f"  95% CI for Kelly        [{k_low:.1f}x, {k_high:.1f}x]")

    levels = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    print(f"\n  OUTCOME DISTRIBUTION over {args.horizon} days "
          f"({args.paths:,} bootstrapped paths)")
    print(f"  {'lev':>5s}{'median':>9s}{'P(2x+)':>9s}{'P(up)':>8s}"
          f"{'P(half)':>9s}{'P(ruin)':>9s}{'growth':>9s}")
    print("  " + "-" * 59)
    for level in levels:
        d = outcome_distribution(returns, level, horizon_days=args.horizon,
                                 num_paths=args.paths)
        print(f"  {level:4.1f}x{d['median']:9.2f}{d['p_double']:9.1%}{d['p_up']:8.1%}"
              f"{d['p_lose_half']:9.1%}{d['p_ruin']:9.1%}{d['mean_log_growth'] * 100:8.1f}%")

    if args.control:
        print("\n  CONTROL: identical risk, edge removed (the scenario you cannot rule out)")
        control = zero_edge_control(returns)
        print(f"  {'lev':>5s}{'median':>9s}{'P(2x+)':>9s}{'P(up)':>8s}"
              f"{'P(half)':>9s}{'P(ruin)':>9s}{'growth':>9s}")
        print("  " + "-" * 59)
        for level in levels:
            d = outcome_distribution(control, level, horizon_days=args.horizon,
                                     num_paths=args.paths)
            print(f"  {level:4.1f}x{d['median']:9.2f}{d['p_double']:9.1%}{d['p_up']:8.1%}"
                  f"{d['p_lose_half']:9.1%}{d['p_ruin']:9.1%}{d['mean_log_growth'] * 100:8.1f}%")
        print("\n  Note that leverage still buys a meaningful chance of doubling even with")
        print("  NO edge at all. A high probability of a big win is therefore not evidence")
        print("  that a strategy is good -- it is evidence that it is volatile.")

    print(f"\n  RECOMMENDED LEVERAGE: {profile.recommended_leverage():.2f}x")
    if not profile.edge_is_established:
        print("  The Sharpe confidence interval includes zero, so the edge is not")
        print("  established. Kelly on an unproven edge is not a small bet -- it is a")
        print("  bet whose sign is unknown. The growth-optimal size is zero.")
    else:
        print(f"  (quarter-Kelly. Full Kelly assumes you know your edge exactly;")
        print("   overestimating it by 2x means betting 2x Kelly, which grows at zero.)")
    print()
    return 0

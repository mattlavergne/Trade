"""Statistical validation: the machinery that stops you fooling yourself.

Backtesting is an unusually effective way to generate false confidence. Try
enough strategies and parameter sets and some will look excellent on any
dataset, including pure noise. The tools here quantify that problem instead of
ignoring it.

Three defences, in increasing order of severity:

1. **Walk-forward analysis.** Fit on a window, trade the next window, roll
   forward. Every trade is made on data the fit never saw.
2. **Probabilistic / Deflated Sharpe Ratio** (Bailey & Lopez de Prado, 2014).
   A Sharpe ratio selected as the best of N attempts is biased upward. The
   Deflated Sharpe corrects for how many attempts it took to find it, and for
   skew and fat tails. It routinely turns an "excellent" Sharpe of 1.5 into a
   statistically worthless one.
3. **Block bootstrap.** Resample contiguous blocks of returns to ask: how often
   would this result appear by chance, preserving autocorrelation?
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: pd.Series, periods_per_year: int = 365) -> float:
    if len(returns) < 2 or returns.std(ddof=1) == 0:
        return 0.0
    return float(returns.mean() / returns.std(ddof=1) * math.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    returns: pd.Series,
    benchmark_sharpe: float = 0.0,
    periods_per_year: int = 365,
) -> float:
    """P(true Sharpe > benchmark), accounting for sample length, skew, kurtosis.

    Returns a probability in [0, 1]. Below ~0.95 you have not demonstrated an
    edge at conventional confidence, however pretty the equity curve is.
    """
    n = len(returns)
    if n < 3:
        return 0.0
    observed = sharpe_ratio(returns, periods_per_year) / math.sqrt(periods_per_year)
    benchmark = benchmark_sharpe / math.sqrt(periods_per_year)
    skew = float(stats.skew(returns, bias=False))
    kurt = float(stats.kurtosis(returns, fisher=False, bias=False))

    denominator = 1.0 - skew * observed + (kurt - 1.0) / 4.0 * observed**2
    if denominator <= 0:
        return 0.0
    statistic = (observed - benchmark) * math.sqrt(n - 1) / math.sqrt(denominator)
    return float(stats.norm.cdf(statistic))


def expected_max_sharpe(num_trials: int, sharpe_variance: float) -> float:
    """Expected maximum Sharpe from `num_trials` independent attempts on noise.

    This is the bar a real edge must clear. Test 50 strategies on random data
    and the best will show a respectable Sharpe purely by chance; this says how
    respectable.
    """
    if num_trials < 2:
        return 0.0
    std = math.sqrt(max(sharpe_variance, 1e-12))
    a = stats.norm.ppf(1.0 - 1.0 / num_trials)
    b = stats.norm.ppf(1.0 - 1.0 / (num_trials * math.e))
    return float(std * ((1.0 - EULER_MASCHERONI) * a + EULER_MASCHERONI * b))


def deflated_sharpe_ratio(
    returns: pd.Series,
    num_trials: int,
    sharpe_variance: float | None = None,
    periods_per_year: int = 365,
) -> float:
    """Probability the strategy's true Sharpe exceeds zero, after correcting
    for the number of configurations tried (selection bias).

    Args:
        num_trials: How many strategy/parameter combinations were evaluated
            before this one was chosen. Be honest here -- undercounting is the
            easiest way to fake significance.
        sharpe_variance: Variance of Sharpe ratios across those trials. If
            unknown, a conservative default is assumed.
    """
    if sharpe_variance is None:
        sharpe_variance = 0.5 / periods_per_year
    threshold = expected_max_sharpe(num_trials, sharpe_variance) * math.sqrt(periods_per_year)
    return probabilistic_sharpe_ratio(returns, threshold, periods_per_year)


def block_bootstrap_pvalue(
    returns: pd.Series,
    num_samples: int = 2000,
    block_size: int = 20,
    seed: int = 0,
) -> float:
    """P-value for "mean return > 0" using a circular block bootstrap.

    Blocks preserve autocorrelation and volatility clustering, which an
    ordinary i.i.d. bootstrap destroys (and thereby understates uncertainty).
    """
    values = returns.dropna().to_numpy()
    n = len(values)
    if n < block_size * 2:
        return 1.0

    rng = np.random.default_rng(seed)
    centred = values - values.mean()  # impose the null: true mean is zero
    observed = values.mean()
    num_blocks = int(np.ceil(n / block_size))

    exceed = 0
    for _ in range(num_samples):
        starts = rng.integers(0, n, size=num_blocks)
        sample = np.concatenate([
            np.take(centred, range(s, s + block_size), mode="wrap") for s in starts
        ])[:n]
        if sample.mean() >= observed:
            exceed += 1
    return float((exceed + 1) / (num_samples + 1))


@dataclass
class WalkForwardWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict
    in_sample_sharpe: float
    out_of_sample_sharpe: float
    out_of_sample_return_pct: float


@dataclass
class WalkForwardResult:
    windows: list[WalkForwardWindow]
    combined_returns: pd.Series
    num_trials: int

    @property
    def out_of_sample_sharpe(self) -> float:
        return sharpe_ratio(self.combined_returns)

    @property
    def in_sample_mean_sharpe(self) -> float:
        return float(np.mean([w.in_sample_sharpe for w in self.windows])) if self.windows else 0.0

    @property
    def degradation(self) -> float:
        """In-sample Sharpe minus out-of-sample Sharpe.

        Large positive degradation is the signature of overfitting: the strategy
        learned the training data's noise rather than a repeatable pattern.
        """
        return self.in_sample_mean_sharpe - self.out_of_sample_sharpe

    @property
    def consistency(self) -> float:
        """Fraction of out-of-sample windows that were profitable."""
        if not self.windows:
            return 0.0
        return float(np.mean([w.out_of_sample_return_pct > 0 for w in self.windows]))

    def verdict(self) -> list[str]:
        """Plain-language findings. Returns a list of warnings; empty is good."""
        issues: list[str] = []
        if self.out_of_sample_sharpe <= 0:
            issues.append(
                f"Out-of-sample Sharpe is {self.out_of_sample_sharpe:.2f} (<= 0). The strategy "
                "did not work on data it had not seen. This is disqualifying."
            )
        if self.degradation > 0.5:
            issues.append(
                f"Sharpe degraded by {self.degradation:.2f} from in-sample to out-of-sample. "
                "That gap is the signature of overfitting."
            )
        if self.consistency < 0.5:
            issues.append(
                f"Only {self.consistency:.0%} of out-of-sample windows were profitable. "
                "Results depend on which period you happened to test."
            )
        psr = probabilistic_sharpe_ratio(self.combined_returns)
        if psr < 0.95:
            issues.append(
                f"Probabilistic Sharpe is {psr:.1%} (< 95%). The out-of-sample record is not "
                "long enough to distinguish this edge from luck."
            )
        dsr = deflated_sharpe_ratio(self.combined_returns, max(self.num_trials, 1))
        if dsr < 0.95:
            issues.append(
                f"Deflated Sharpe is {dsr:.1%} after correcting for {self.num_trials} "
                "configurations tested. Selection bias alone could explain this result."
            )
        return issues


def walk_forward(
    panel,
    strategy_factory,
    param_grid: list[dict],
    venue,
    *,
    config=None,
    train_bars: int = 365,
    test_bars: int = 90,
    step_bars: int | None = None,
    starting_cash: float = 10_000.0,
    warmup: int = 100,
) -> WalkForwardResult:
    """Rolling walk-forward: optimise on train, trade on test, roll forward.

    The critical property: parameters used in each test window were chosen using
    only data from before that window. Nothing is fitted on data it then trades.
    """
    from .portfolio import run_portfolio_backtest

    step_bars = step_bars or test_bars
    windows: list[WalkForwardWindow] = []
    oos_returns: list[pd.Series] = []

    start = 0
    while start + train_bars + test_bars <= len(panel):
        train_end = start + train_bars
        test_end = min(train_end + test_bars, len(panel))

        train_panel = panel.slice(train_end)
        best_params, best_sharpe = None, -np.inf

        for params in param_grid:
            try:
                result = run_portfolio_backtest(
                    train_panel, strategy_factory(**params), venue,
                    config=config, starting_cash=starting_cash, warmup=warmup,
                )
                candidate = sharpe_ratio(result.daily_returns())
            except Exception:  # noqa: BLE001 - a bad param set must not abort the sweep
                continue
            if candidate > best_sharpe:
                best_sharpe, best_params = candidate, params

        if best_params is None:
            start += step_bars
            continue

        # Trade the test window with the chosen parameters. The panel includes
        # history so indicators are warm, but only test-window returns count.
        test_panel = panel.slice(test_end)
        test_result = run_portfolio_backtest(
            test_panel, strategy_factory(**best_params), venue,
            config=config, starting_cash=starting_cash, warmup=warmup,
        )
        window_returns = test_result.daily_returns().iloc[-(test_end - train_end):]

        if len(window_returns) > 1:
            oos_returns.append(window_returns)
            windows.append(WalkForwardWindow(
                train_start=panel.index[start],
                train_end=panel.index[train_end - 1],
                test_start=panel.index[train_end],
                test_end=panel.index[test_end - 1],
                best_params=best_params,
                in_sample_sharpe=best_sharpe,
                out_of_sample_sharpe=sharpe_ratio(window_returns),
                out_of_sample_return_pct=float(
                    (1.0 + window_returns).prod() - 1.0
                ) * 100.0,
            ))

        start += step_bars

    combined = pd.concat(oos_returns) if oos_returns else pd.Series(dtype=float)
    return WalkForwardResult(
        windows=windows,
        combined_returns=combined,
        num_trials=len(param_grid) * max(len(windows), 1),
    )

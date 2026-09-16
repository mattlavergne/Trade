# Trade — a systematic trading research system

A framework for building trading strategies and, more importantly, for finding
out whether they actually work before any money is at risk.

It contains a multi-asset portfolio engine with volatility targeting, walk-forward
validation with multiple-testing correction, realistic cost and funding models,
and live paper trading. **No code here can place a real order** — there is no
credential handling and no private endpoint.

**124 tests**, including automated look-ahead-bias detection.

---

## What the system actually found

Every number below came out of this repo on real market data. They are
reproducible with the commands shown.

### 1. Risk management is the edge. Signals mostly are not.

`python cli.py portfolio` — 14 crypto assets, 4.1 years of daily bars:

| | Vol-targeted hold | Raw buy & hold |
|---|---:|---:|
| CAGR | **11.9%** | 10.3% |
| Volatility | **21.8%** | 63.1% |
| Sharpe | **0.63** | 0.47 |
| Max drawdown | **−29.8%** | −75.4% |
| Calmar | **0.40** | 0.14 |
| Costs | 0.7% | 0% |

Both rows hold *the same assets in the same proportions*. The only difference is
that one sizes positions inversely to recent volatility. That single change cut
drawdown by two-thirds and raised Sharpe by a third, with almost no turnover.

The forecasting strategies — time-series momentum, cross-sectional momentum,
trend-filtered momentum — all generated 10–19x more turnover, paid 22–50% of
capital in fees, and **none survived out-of-sample validation**.

### 2. The out-of-sample test that kills most strategies

`python cli.py walkforward --strategy trend_filtered`:

| Metric | Value |
|---|---:|
| In-sample Sharpe | **1.12** |
| Out-of-sample Sharpe | **0.54** |
| Degradation | 0.58 |
| Windows profitable | **44%** |
| Bootstrap p-value | 0.222 |
| **Deflated Sharpe** | **1.5%** |

An in-sample Sharpe of 1.12 looks deployable. After walk-forward and correcting
for the 108 configurations tested, the probability the edge is real is **1.5%**.
Fewer than half the out-of-sample windows were profitable.

### 3. Why your backtest is probably noise

`python -c "from trader.validation import expected_max_sharpe; ..."`

| Strategies tried | Expected best Sharpe **on pure random noise** |
|---:|---:|
| 10 | 1.11 |
| 50 | **1.61** |
| 200 | 1.96 |
| 1000 | 2.30 |

Test 50 variants on random data and the best will show a Sharpe of 1.61 — a
figure most people would consider excellent and deploy immediately. This is why
a backtest without a multiple-testing correction is worthless, and why the
Deflated Sharpe Ratio is applied everywhere in this repo.

### 4. Crypto diversification is mostly an illusion

`python cli.py universe`:

```
  Mean pairwise correlation: 0.66
  Effective independent bets: 1.5 (of 14 assets)
```

Holding 14 crypto assets is closer to holding **1.5**. The "scan 50 markets"
pitch fails on this alone, before fees are even considered. Real diversification
requires assets that are genuinely different — equities, bonds, commodities,
currencies — not fourteen flavours of the same beta.

### 5. Carry is a real edge, and still too small to matter here

`python cli.py funding` — OKX perpetual funding, 93 days:

| Asset | Funding annualized | % periods positive |
|---|---:|---:|
| DOGE | +6.18% | 87.2% |
| BTC | +5.04% | 89.0% |
| ETH | +3.74% | 78.6% |

Longs pay shorts ~85% of the time, because retail leverage demand is
structurally long. Collecting it (long spot + short perp) is **market-neutral
income, not a forecast**. Then reality:

```
  expected gross (after 50% capital efficiency):  +2.52%/yr
  cost drag over a 94-day hold:                   -1.56%/yr
  EXPECTED NET:                                   +0.96%/yr

  basis-noise std across 400 draws:               10.02%/yr
  P(profitable over this window):                 56%
```

The edge is real and positive. It is also **1/10th the size of the basis noise**
around it. At a Sharpe of 0.096, demonstrating it statistically would take
roughly **436 years** of data. It is an institutional trade that works on size
and patience, and US persons cannot legally access the venues where it works.

### 6. More risk buys a bigger win and a *lower* win rate

`python cli.py leverage --control`

Leverage multiplies arithmetic return linearly but volatility drag
**quadratically**, so growth rises, peaks, and then falls:

| Leverage | Median | P(double) | **P(any gain)** | P(lose half) | Growth |
|---:|---:|---:|---:|---:|---:|
| 1x | 1.10 | 1.3% | **65.6%** | 0.0% | +10.3% |
| 3x | 1.15 | 22.5% | **58.4%** | 11.5% | +16.6% |
| 5x | 1.00 | 28.5% | **50.0%** | 28.1% | +3.6% |

Read the middle column. Leverage raises your chance of a *big* win (1.3% to
28.5%) while lowering your chance of *any* win (65.6% to 50.0%). "Win more often
than I lose" and "swing for a big score" are opposite requests, and leverage
trades one for the other.

Past Kelly (here 2.68x) it gets strictly worse: at 2x Kelly growth is exactly
zero with double the volatility. At 6x, arithmetic return is +81.7%/yr while
capital compounds at **−3.8%/yr**.

And the control column is the point:

| Leverage | P(double) **with no edge** | P(lose half) | Growth |
|---:|---:|---:|---:|
| 3x | 10.7% | 26.3% | **−21.5%** |
| 5x | 14.6% | 47.7% | **−60.2%** |

**Leverage buys a meaningful chance of doubling even when expected value is
deeply negative.** A high probability of a big win is not evidence a strategy is
good — it is evidence it is volatile. Volatility is free; you do not need
software to obtain it.

Because the measured Sharpe here is 0.63 ± 0.54 — a confidence interval that
includes zero — the tool recommends **0.00x**. Kelly sizing on an unproven edge
is not a small bet; it is a bet whose sign is unknown.

---

## So what should someone actually do?

The evidence in this repo points one direction, and it is not exciting:

1. **Own a diversified portfolio.** Not 14 correlated crypto assets — genuinely
   different asset classes.
2. **Size positions by volatility, not by conviction.** This is the one change
   that reliably improved outcomes in every test here, and it requires
   predicting nothing.
3. **Cap drawdowns mechanically.** The circuit breaker in `trader/risk.py` cuts
   exposure as losses deepen. It costs some upside and prevents ruin.
4. **Trade as rarely as you can stand.** Widening the no-trade band improved
   every strategy monotonically. Turnover is a certain cost against an uncertain
   benefit.
5. **Assume your signal is noise until walk-forward says otherwise.** It usually
   is.

That is a genuinely effective system. It is also, deliberately, closer to
"disciplined investing" than to a bot that scalps arbitrage — because that is
what the data supports.

**On account size:** none of this makes $50 grow meaningfully. A great 15%/year
on $50 is $7.50. The machinery here is worth building because it is transferable
and it prevents expensive mistakes, not because $50 will compound into anything.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Crypto market data needs no credentials. Equity data via Alpaca needs free paper
keys from <https://alpaca.markets>.

## Usage

**Portfolio system (the serious one):**

```bash
python cli.py universe          # assets, correlations, effective bets
python cli.py portfolio         # multi-asset vol-targeted backtest
python cli.py walkforward       # out-of-sample validation + overfitting checks
python cli.py funding           # perpetual funding and carry economics
python cli.py leverage --control # how much risk is justified, and its ruin cost
```

**Single-asset system (simpler, good for learning):**

```bash
python cli.py venues            # fee schedules and breakeven hurdles
python cli.py spread-check      # live cross-venue spread vs the fee floor
python cli.py compare           # every strategy side by side
python cli.py backtest --strategy donchian --html
python cli.py paper --strategy donchian --poll 60    # live paper trading
python cli.py status
```

Useful knobs on the portfolio commands:

```bash
python cli.py portfolio --target-vol 0.20 --band 0.75 --venue kraken
python cli.py walkforward --strategy tsmom --train 500 --test 150
```

---

## How this avoids lying to you

Backtesting is an unusually effective way to generate false confidence. Every
defence below is enforced by a test that fails loudly if it breaks.

**One bar of latency, always.** Strategies see data through bar *t* and execute
at bar *t+1*'s open. `tests/test_portfolio.py` includes an oracle strategy that
signals *tomorrow's* return; if the engine ever lets it compound, the test fails
and every result in the repo is void.

**Causality checks.** `tests/test_strategies.py` truncates future bars and
asserts no past signal changes. A strategy reading the future fails this.

**Costs are itemised, never buried.** Fees and slippage are tracked separately
from PnL. Minimum notional is enforced by *rejection*, not silent resizing.

**The benchmark pays the same costs.** Buy-and-hold runs through the same broker
and fee schedule, so comparisons are honest in both directions.

**Multiple-testing correction everywhere.** The Deflated Sharpe Ratio (Bailey &
López de Prado) corrects for how many configurations were tried. Walk-forward
reports in-sample vs out-of-sample degradation explicitly.

**Metrics refuse to overclaim.** Fewer than ~30 trades, |t| < 2, samples under
six months, or cost drag over 10% all produce explicit warnings instead of
confident-looking numbers.

**The fast path is proven equivalent.** The numpy hot loop is checked against the
readable pandas reference across 24 randomised cases in
`tests/test_risk_fastpath.py`.

---

## Layout

```
trader/
  costs.py         venue fee schedules, breakeven arithmetic
  data.py          ccxt OHLCV fetching with on-disk cache
  universe.py      multi-asset aligned price panels
  broker.py        paper broker: fills, fees, slippage, rejections
  risk.py          vol targeting, drawdown control, no-trade bands (+numpy)
  portfolio.py     multi-asset backtest engine
  engine.py        single-asset backtest engine
  validation.py    walk-forward, deflated Sharpe, block bootstrap
  leverage.py      Kelly sizing, volatility drag, ruin probabilities
  funding.py       perpetual funding rates and carry economics
  metrics.py       performance stats with significance warnings
  live.py          live paper trading, crash-safe state
  report.py        console output and self-contained HTML reports
  strategy.py      single-asset strategy interface
  strategies/      single-asset + portfolio strategies
cli.py             command line interface
tests/             124 tests incl. look-ahead and causality detection
```

## Writing a portfolio strategy

```python
from trader.portfolio import PortfolioStrategy

class MyIdea(PortfolioStrategy):
    name = "my_idea"

    def __init__(self, lookback: int = 60) -> None:
        self.lookback = lookback

    def signals(self, panel):
        # Return a DataFrame (date x symbol) in [-1, 1].
        # Express DIRECTION AND CONVICTION only -- position sizing is the
        # engine's job, via volatility targeting.
        # Row t may use ONLY data through row t.
        momentum = panel.close.pct_change(self.lookback)
        return momentum.apply(lambda c: c.clip(-1, 1)).fillna(0.0)
```

Register it in `PORTFOLIO_STRATEGIES` and it appears in the CLI. Then run
`walkforward` on it before believing anything it tells you.

---

## Hosting

The paper trader is a long-running process, so shared web hosting will not run
it. Use a local machine or a small VPS under `systemd` or `tmux`.

`reports/*.html` are fully self-contained — no external CSS, JS or fonts, inline
SVG charts, light and dark themes. Upload anywhere static and they work.

---

## Before risking real money

1. Backtest across **multiple distinct date ranges**, including a bear market.
2. Run `walkforward`. If the deflated Sharpe is below 95%, stop.
3. Paper trade for **at least a month** on data that did not exist when the
   strategy was written. Most strategies die here, for free.
4. Confirm paper results **match** backtest expectations. A large gap means the
   backtest was wrong.
5. Only then consider real money, in an amount whose total loss is irrelevant
   to you.

## What this repo will never tell you

That it found a strategy with high confidence of not losing money. No such thing
exists at any account size. Anyone offering you one — in a repo, a course, or a
viral post with an auto-DM funnel attached — is selling something.

The most valuable output here is negative results, arrived at honestly and
cheaply.

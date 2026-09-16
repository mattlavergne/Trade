# Trade — a strategy research harness

A backtesting and live paper-trading framework for finding out whether a trading
strategy has an edge **before** any money is at risk.

This repository does not contain a profitable trading bot. It contains the
instrument you use to discover whether one is possible. That distinction is the
entire point, and the rest of this document explains why.

**No code in this repository can place a real order.** There is no API key
handling, no request signing, no private endpoint. It is structurally incapable
of spending money.

---

## Why this repo exists, and what it replaces

This was started after reading a viral post claiming a student turned **$68 into
$750,000** with a Claude Code trading bot, including **$6,732 profit on the first
night**, by "scanning 50 markets" for "price errors."

That story is arithmetically impossible, and the tools in this repo let you
verify that yourself rather than take anyone's word for it.

### The internal contradiction

The post attributes the gains to *arbitrage* — "spots price errors," "no
guessing," "mispricing across dozens of markets." Arbitrage is the
**lowest-variance** strategy that exists; it produces small, boring, consistent
returns. A 99x overnight return requires ~100x leverage on a single directional
bet, which is the **highest-variance** thing you can do.

The post describes a risk-free strategy delivering lottery-ticket returns. Those
are mutually exclusive. That alone is disqualifying.

### Check the arbitrage claim against live data

```bash
python cli.py spread-check
```

A real run, 2026-09-16:

```
  coinbase     $75,793.75
  binanceus    $75,831.27
  kraken       $75,831.80

  Widest gap:      buy coinbase / sell kraken
  Gross spread:    $38.05  (5.02 bps)
  Taker fees:      160.0 bps  (120 buy + 40 sell)
  Net edge:        -154.98 bps
```

The widest cross-venue gap available was **5.02 basis points**. Capturing it
costs **160 basis points** in taker fees. You are **32x underwater** before
accounting for the fact that the gap closes in milliseconds and you would be
racing firms with colocated hardware while polling once per second.

This is the normal state of the market, not an unlucky moment. Run it yourself.

---

## The finding that matters more than any strategy

Run `python cli.py compare` and you get the central result of this project.
Here it is on 9,591 hourly BTC/USD bars (Aug 2025 – Sep 2026, $50 start):

| Strategy | Zero fees | Kraken fees | Coinbase fees |
|---|---:|---:|---:|
| buy_and_hold | −36.92% | −37.17% | −37.67% |
| sma_cross | **−1.77%** | −56.75% | **−91.35%** |
| mean_reversion | −29.04% | −73.59% | −96.20% |
| donchian | **−7.51%** | −46.62% | −81.85% |

Read that table twice.

**With zero fees, every active strategy beats buy-and-hold.** `sma_cross` turns a
−36.92% market into −1.77%. They look like they work. This is the backtest that
gets screenshotted and posted.

**With real fees, every single one flips to worse than doing nothing.** On
Coinbase, `sma_cross` paid **$41.28 in fees on a $50 account** — 82% of the
capital handed to the exchange — and ended at $4.33.

The strategies did not change. Only the cost model did. **Fees, not signal
quality, determine the outcome at small account sizes.** Any backtest that
does not model them to the basis point is fiction.

---

## Why $50 is the real problem

This is market-independent arithmetic, and no choice of instrument fixes it.

Sound risk management means risking **1–2% of the account per trade**. On $50
that is **$0.50–$1.00 of risk per trade**.

```
python cli.py venues
```

```
    coinbase           $50 round trip costs $ 1.22   UNRUNNABLE
    binanceus          $50 round trip costs $ 0.61   UNRUNNABLE
    kraken             $50 round trip costs $ 0.41   viable
    alpaca_equities    $50 round trip costs $ 0.02   viable
```

On Coinbase your **transaction cost exceeds your entire risk budget**. You would
be paying more to open the position than you are willing to lose on it. No
strategy is clever enough to overcome that, because it is not a strategy problem.

The honest summary: **$50 is below the threshold where trading returns can matter.**
Even a genuinely excellent 15%/year on $50 is **$7.50 a year**. The skill you
build reading this repo is worth vastly more than the capital at stake.

---

## On forex (asked, and answered honestly)

Retail FX looks cheap — `retail_fx` shows a 0.07% breakeven versus Coinbase's
2.43%. That is real, and it is also a trap. Three reasons FX is **worse**, not
better, for a small account:

1. **Leverage converts small moves into total loss.** US regulation caps retail
   FX at 50:1 on majors. At 50:1, a **2% adverse move wipes out 100% of the
   account**. EUR/USD routinely moves 0.5–0.8% per day. You are two or three
   ordinary days from zero. The low spread is what *lures* you into the leverage
   that kills you.
2. **No positive drift to fall back on.** Equities and crypto trend upward over
   long horizons, so "hold and wait" is a genuine fallback. Currency pairs are
   approximately zero-sum and mean-reverting — EUR/USD is not systematically
   higher today than in 2000. You are **100% dependent on having an edge**,
   against bank desks that see order flow you never will.
3. **Brokers' own disclosures** put roughly **70–75% of retail FX accounts at a
   loss**. That is their regulatory filing, not an outside critic's estimate.

The harness includes a `retail_fx` cost profile so you can model it, but the
spread is not what destroys FX accounts. Leverage is, and leverage is not a
modelling problem — it is a ruin problem.

---

## The one genuine improvement: commission-free equities

`alpaca_equities` requires a **0.04%** move to break even versus Coinbase's
**2.43%** — a **60x lower hurdle**. Alpaca charges no commission, supports
fractional shares (so $50 works), and offers a free paper-trading API.

This does not make you profitable. It removes the single largest structural
reason you would be guaranteed *un*profitable. It is the difference between a
losing game and a fair-ish one.

Two constraints to understand before relying on it:

- **Pattern Day Trader rule.** Under $25,000 equity you get **3 day trades per
  rolling 5 business days**. Strategies that round-trip intraday are simply not
  runnable. This pushes you toward lower-frequency strategies — which, given the
  fee analysis above, you should want anyway.
- **You still pay the spread.** Commission-free is not cost-free.

---

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

Crypto market data needs no credentials. Equity data via Alpaca needs free paper
keys from <https://alpaca.markets> (`ALPACA_API_KEY` / `ALPACA_SECRET_KEY`).

## Usage

```bash
python cli.py venues                                  # cost table and hurdle rates
python cli.py spread-check                            # live arbitrage reality check
python cli.py compare --venue kraken                  # all strategies, side by side
python cli.py backtest --strategy donchian --html     # one strategy + HTML report
python cli.py paper --strategy donchian --poll 60     # live paper trading
python cli.py status                                  # paper session state
```

Tune strategy parameters with repeated `--param`:

```bash
python cli.py backtest --strategy sma_cross --param fast=10 --param slow=40
```

## How the backtest avoids lying to you

Four design decisions, each guarding against a specific way backtests flatter
losing strategies:

1. **One bar of latency, always.** The strategy sees bars up to and including
   bar *t* and emits a target exposure; the broker rebalances at bar *t+1*'s
   **open**. Filling on the signal bar's close is look-ahead bias and is the
   most common source of imaginary profit. `tests/test_engine.py` includes an
   oracle strategy that deliberately cheats; if it ever profits, the test fails
   and every result in the repo is void.
2. **Costs are explicit and itemised.** Every fill records its fee and slippage
   separately from PnL, so cost drag is visible rather than buried in returns.
3. **Minimum notional is enforced by rejection, not by silent resizing.** At $50
   this rejects real trades. That is accurate, and it is information.
4. **The benchmark pays the same costs.** Buy-and-hold is run through the same
   broker and fee schedule, so the comparison is honest in both directions.

Additionally, `tests/test_strategies.py` verifies **causality** for every
strategy: truncating future bars must not change any past signal value. A
strategy that fails this is reading the future.

## Metrics, with the caveats attached

The report refuses to print a confident number where none is warranted:

- Fewer than ~30 round trips → per-trade statistics flagged as **noise**.
- Mean trade PnL with |t| < 2 → flagged as **not distinguishable from zero**.
- Sample under 6 months → annualised figures flagged as **meaningless**.
- Cost drag over 10% of capital → flagged as **trading too often for this size**.
- Any rejected orders → flagged as **not fully executable at this account size**.

## Layout

```
trader/
  costs.py       venue fee schedules, breakeven arithmetic
  data.py        ccxt OHLCV fetching with on-disk cache
  broker.py      paper broker: fills, fees, slippage, rejections
  strategy.py    strategy interface and registry
  strategies/    buy_and_hold, sma_cross, mean_reversion, donchian
  engine.py      backtest loop (the no-look-ahead guarantee lives here)
  metrics.py     performance stats + statistical-significance warnings
  live.py        live paper trading with crash-safe state persistence
  report.py      console output and self-contained HTML reports
cli.py           command line interface
tests/           42 tests, including look-ahead and causality detection
```

## Hosting

The bot is a long-running process, so shared web hosting will not run it. Run
`cli.py paper` on a local machine or a small VPS under `systemd` or `tmux`.

`reports/*.html` are **fully self-contained** — no external CSS, JS, or fonts,
inline SVG charts, light and dark themes. Upload one anywhere static (including
mattlavergne.com) and it works.

## Writing your own strategy

```python
# trader/strategies/my_idea.py
import pandas as pd
from ..strategy import Strategy, register

@register("my_idea")
class MyIdea(Strategy):
    def __init__(self, lookback: int = 20) -> None:
        self.lookback = lookback

    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        # Return a Series in [0, 1] aligned to bars.index.
        # The value at bar t may use ONLY data up to and including bar t.
        momentum = bars["close"].pct_change(self.lookback)
        return self._clip((momentum > 0).astype(float))
```

Import it in `trader/strategies/__init__.py` and it appears in the CLI
automatically. The causality test in `tests/test_strategies.py` will check it for
look-ahead bias — add its name to `BUILT_IN` there.

---

## Before you consider risking real money

A checklist, in order. Skipping steps is how people lose money.

1. Backtest over **multiple distinct date ranges**, including a bear market.
   Beating buy-and-hold on one sample is weak evidence.
2. Confirm the edge is **statistically significant** — 30+ round trips and
   |t| > 2 on mean trade PnL. The harness tells you when it is not.
3. Paper trade live for **at least a month** on data that did not exist when the
   strategy was written. Most strategies die here, for free.
4. Confirm paper results **match** backtest expectations. A large gap means the
   backtest was wrong, not that the market was unusual.
5. Only then consider real money — and start with an amount whose total loss is
   genuinely irrelevant to you.

If a strategy cannot make money on paper, where there is no slippage beyond the
model, no rejected orders, no downtime and no emotion, it will not make money
live.

## What this repo will never tell you

That it found a strategy with a high level of confidence of not losing money. No
such thing exists at any account size, and anyone offering you one — in a repo,
a course, or a viral post with an auto-DM funnel attached — is selling something.

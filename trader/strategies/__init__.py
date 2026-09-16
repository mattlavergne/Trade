"""Built-in strategies.

None of these is expected to be profitable net of fees. They are reference
implementations and honest baselines -- the harness exists to measure ideas,
not to supply a winning one.
"""

from . import buy_and_hold, donchian, mean_reversion, sma_cross  # noqa: F401

__all__ = ["buy_and_hold", "donchian", "mean_reversion", "sma_cross"]

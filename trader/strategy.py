"""Strategy interface and registry.

A strategy's only job is to turn a price history into a target exposure in
[0, 1], where 0 is fully in cash and 1 is fully invested. It never sees the
broker, the cash balance, or its own PnL -- that separation is what stops a
strategy from accidentally peeking at information it would not have live.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import pandas as pd

_REGISTRY: dict[str, Callable[..., "Strategy"]] = {}


def register(name: str) -> Callable[[type["Strategy"]], type["Strategy"]]:
    def decorator(cls: type["Strategy"]) -> type["Strategy"]:
        _REGISTRY[name] = cls
        cls.name = name
        return cls

    return decorator


def get_strategy(name: str, **kwargs) -> "Strategy":
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


class Strategy(ABC):
    """Base strategy.

    Subclasses implement `target_exposure`, returning a Series aligned to the
    input index with values in [0, 1].

    CRITICAL: the value at index t must be computable using only data up to and
    including bar t. The engine executes it at bar t+1's open. Any use of
    future data here silently fabricates profit.
    """

    name: str = "unnamed"

    @abstractmethod
    def target_exposure(self, bars: pd.DataFrame) -> pd.Series:
        """Map OHLCV bars to a target exposure in [0, 1]."""

    def describe(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in sorted(vars(self).items()))
        return f"{self.name}({params})"

    @staticmethod
    def _clip(series: pd.Series) -> pd.Series:
        return series.clip(lower=0.0, upper=1.0).fillna(0.0)

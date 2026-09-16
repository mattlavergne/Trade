import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def flat_bars() -> pd.DataFrame:
    """200 bars at a constant price. Any PnL here is pure cost."""
    index = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")
    price = 100.0
    return pd.DataFrame(
        {"open": price, "high": price, "low": price, "close": price, "volume": 1.0},
        index=index,
    )


@pytest.fixture
def trending_bars() -> pd.DataFrame:
    """200 bars rising steadily, so trend strategies have something to find."""
    index = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")
    close = np.linspace(100.0, 200.0, 200)
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1.0},
        index=index,
    )


@pytest.fixture
def spike_bars() -> pd.DataFrame:
    """Flat, then a single enormous one-bar spike at bar 150.

    A strategy with look-ahead bias will capture this spike. An honest one
    cannot, because the spike is not knowable until after it has happened.
    """
    index = pd.date_range("2025-01-01", periods=200, freq="1h", tz="UTC")
    close = np.full(200, 100.0)
    close[150] = 1000.0
    return pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1.0},
        index=index,
    )

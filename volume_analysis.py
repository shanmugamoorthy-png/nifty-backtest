"""Volume confirmation without volume indicators: relative volume, its class, high-volume rejection, price-volume divergence.
Volume here = NIFTY 50 stocks' traded value in each candle (the index itself has no volume)."""
import numpy as np
import pandas as pd

from indicators import candle_parts


def relative_volume(bars, lookback=20, mode="candles"):
    """Candle volume / average volume. mode 'candles' = previous `lookback` candles (as in the spec);
    'time' = the same time of day over the previous `lookback` days (fairer: 09:15 is always busy)."""
    v = bars.v.astype(float)
    if mode == "time":
        avg = v.groupby(bars.mins).transform(lambda s: s.shift(1).rolling(lookback, min_periods=5).mean())
    else:
        avg = v.shift(1).rolling(lookback, min_periods=5).mean()
    return (v / avg.replace(0, np.nan)).to_numpy()


def volume_class(rv, strong=1.5, moderate=1.2):
    """2 = strong (above `strong` x average), 1 = moderate, 0 = weak, -1 = no volume data."""
    rv = np.asarray(rv, float)
    out = np.where(rv >= strong, 2, np.where(rv >= moderate, 1, 0))
    return np.where(np.isfinite(rv), out, -1)


VOL_NAMES = {2: "strong", 1: "moderate", 0: "weak", -1: "no data"}


def high_volume_rejection(bars, rv, strong=1.5):
    """+1 = buyers rejected lower prices on heavy volume (long lower wick, close in the upper half); -1 = the opposite."""
    body, rng, up, lo = candle_parts(bars.o, bars.h, bars.l, bars.c)
    pos = (bars.c.to_numpy() - bars.l.to_numpy()) / rng
    heavy = np.nan_to_num(rv) >= strong
    return np.where(heavy & (lo >= 0.5 * rng) & (pos >= 0.5), 1, np.where(heavy & (up >= 0.5 * rng) & (pos <= 0.5), -1, 0))


def divergence(bars, rv, n=20):
    """-1 = price makes a new n-candle high on below-average volume (weak rally); +1 = new low on weak volume."""
    c = bars.c
    hi = c >= c.rolling(n, min_periods=n).max()
    lo = c <= c.rolling(n, min_periods=n).min()
    weak = np.nan_to_num(rv, nan=9) < 1.0
    return np.where(hi & weak, -1, np.where(lo & weak, 1, 0))

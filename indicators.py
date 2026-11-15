"""Small price helpers. Every value at candle t uses only candles up to and including t (known at t's close)."""
import numpy as np
import pandas as pd


def atr(h, l, c, n=14):
    """Average true range (simple average of the last n true ranges)."""
    h, l, c = (np.asarray(x, float) for x in (h, l, c))
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum.reduce([h - l, np.abs(h - pc), np.abs(l - pc)])
    return pd.Series(tr).rolling(n, min_periods=3).mean().to_numpy()


def candle_parts(o, h, l, c):
    """Body, full range, upper wick, lower wick of every candle."""
    o, h, l, c = (np.asarray(x, float) for x in (o, h, l, c))
    body = np.abs(c - o)
    rng = np.maximum(h - l, 1e-9)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    return body, rng, upper, lower


def same_day_prev(day):
    """True where the previous candle belongs to the same trading day."""
    d = np.asarray(day)
    return np.concatenate([[False], d[1:] == d[:-1]])


def day_prev_high_low(bars):
    """Previous trading day's high and low for every candle."""
    g = bars.groupby("day").agg(dh=("h", "max"), dl=("l", "min"))
    g[["pdh", "pdl"]] = g[["dh", "dl"]].shift(1)
    m = bars[["day"]].merge(g[["pdh", "pdl"]], left_on="day", right_index=True, how="left")
    return m.pdh.to_numpy(), m.pdl.to_numpy()

"""Candlestick patterns. Each returns +1 (bullish), -1 (bearish) or 0 at the candle where the pattern completes.
Two-candle patterns need both candles on the same day (an overnight gap is not a pattern)."""
import numpy as np
import pandas as pd

from indicators import candle_parts, same_day_prev


def patterns(bars, atr_v):
    o, h, l, c = (bars[x].to_numpy(float) for x in "ohlc")
    body, rng, up, lo = candle_parts(o, h, l, c)
    sd = same_day_prev(bars.day)
    sd2 = sd & np.concatenate([[False], sd[:-1]])
    p = lambda a: np.concatenate([[np.nan], a[:-1]])
    p2 = lambda a: np.concatenate([[np.nan, np.nan], a[:-2]])
    o1, h1, l1, c1, b1 = p(o), p(h), p(l), p(c), p(body)
    h2, l2 = p2(h), p2(l)
    bull, bear = c > o, c < o
    out = {}
    out["engulfing"] = np.where(sd & bull & (c1 < o1) & (o <= c1) & (c >= o1) & (body >= b1), 1,
                                np.where(sd & bear & (c1 > o1) & (o >= c1) & (c <= o1) & (body >= b1), -1, 0))
    out["hammer"] = np.where((lo >= 2 * body) & (up <= 0.3 * rng) & (body > 0), 1, 0)
    out["shooting_star"] = np.where((up >= 2 * body) & (lo <= 0.3 * rng) & (body > 0), -1, 0)
    out["pin_bar"] = np.where(lo >= 2 / 3 * rng, 1, np.where(up >= 2 / 3 * rng, -1, 0))
    inside1 = sd2 & (h1 <= h2) & (l1 >= l2)                       # candle t-1 is inside candle t-2
    out["inside_breakout"] = np.where(inside1 & (c > h1), 1, np.where(inside1 & (c < l1), -1, 0))
    outside = sd & (h > h1) & (l < l1)
    pos = (c - l) / rng
    out["outside_bar"] = np.where(outside & (pos >= 0.7), 1, np.where(outside & (pos <= 0.3), -1, 0))
    strong = (rng >= 1.5 * np.nan_to_num(atr_v, nan=np.inf)) & (body >= 0.7 * rng)
    out["momentum"] = np.where(strong & bull, 1, np.where(strong & bear, -1, 0))
    return pd.DataFrame(out, index=bars.index).fillna(0).astype(int)


PATTERN_POINTS = {"engulfing": 20, "hammer": 18, "shooting_star": 18, "pin_bar": 16, "inside_breakout": 18,
                  "outside_bar": 14, "momentum": 20}


def bullish_candle(bars):
    """A clear bullish close: green with a body of at least 40% of the range. -1 for the bearish mirror."""
    body, rng, up, lo = candle_parts(bars.o, bars.h, bars.l, bars.c)
    c, o = bars.c.to_numpy(), bars.o.to_numpy()
    return np.where((c > o) & (body >= 0.4 * rng), 1, np.where((c < o) & (body >= 0.4 * rng), -1, 0))

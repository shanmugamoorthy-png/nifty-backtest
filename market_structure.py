"""Market structure (swing points, HH/HL/LH/LL, BOS, CHoCH), support/resistance levels and supply/demand zones.
No look-ahead: a swing high at candle i is only known at candle i + k, after k lower highs have closed."""
import numpy as np
import pandas as pd

from indicators import candle_parts


def _pivot_flags(h, l, k):
    n = len(h)
    sh, sl = np.zeros(n, bool), np.zeros(n, bool)
    for i in range(k, n - k):
        if h[i] > h[i - k:i].max() and h[i] >= h[i + 1:i + k + 1].max():
            sh[i] = True
        if l[i] < l[i - k:i].min() and l[i] <= l[i + 1:i + k + 1].min():
            sl[i] = True
    return sh, sl


def structure(bars, k=3):
    """Walk forward candle by candle. Returns per-candle trend (+1 HH+HL, -1 LH+LL, 0 mixed), the last known swing
    high/low, BOS/CHoCH events (+1 BOS up, +2 CHoCH up, -1 BOS down, -2 CHoCH down) and the list of labelled pivots."""
    h, l, c = (bars[x].to_numpy(float) for x in "hlc")
    n = len(h)
    sh, sl = _pivot_flags(h, l, k)
    trend = np.zeros(n, int)
    last_sh, last_sl = np.full(n, np.nan), np.full(n, np.nan)
    event = np.zeros(n, int)
    hs, ls = [], []                      # confirmed swing highs / lows (price)
    pivots = []                          # (pivot index, confirmed at, price, label)
    sh_broken = sl_broken = True
    tr = 0
    for t in range(n):
        i = t - k
        if i >= 0 and sh[i]:
            lab = "HH" if hs and h[i] > hs[-1] else ("LH" if hs else "SH")
            hs.append(h[i])
            pivots.append((i, t, h[i], lab))
            sh_broken = False
        if i >= 0 and sl[i]:
            lab = "HL" if ls and l[i] > ls[-1] else ("LL" if ls else "SL")
            ls.append(l[i])
            pivots.append((i, t, l[i], lab))
            sl_broken = False
        if len(hs) >= 2 and len(ls) >= 2:
            tr = 1 if hs[-1] > hs[-2] and ls[-1] > ls[-2] else (-1 if hs[-1] < hs[-2] and ls[-1] < ls[-2] else 0)
        if hs and not sh_broken and c[t] > hs[-1]:
            event[t] = 2 if tr == -1 else 1
            sh_broken = True
        elif ls and not sl_broken and c[t] < ls[-1]:
            event[t] = -2 if tr == 1 else -1
            sl_broken = True
        trend[t] = tr
        last_sh[t] = hs[-1] if hs else np.nan
        last_sl[t] = ls[-1] if ls else np.nan
    return {"trend": trend, "last_sh": last_sh, "last_sl": last_sl, "event": event,
            "pivots": pd.DataFrame(pivots, columns=["i", "known_at", "price", "label"])}


def htf_trend_on(bars, tf, htf_bars, htf, k=3):
    """Higher-timeframe trend seen by each trading candle: only higher-timeframe candles that have already CLOSED."""
    st = structure(htf_bars, k)
    a = pd.DataFrame({"t": bars.dt + pd.Timedelta(minutes=tf)})
    b = pd.DataFrame({"t": htf_bars.dt + pd.Timedelta(minutes=htf), "trend": st["trend"]}).sort_values("t")
    m = pd.merge_asof(a.reset_index().sort_values("t"), b, on="t", direction="backward").sort_values("index")
    return m.trend.fillna(0).astype(int).to_numpy()


def sr_levels(bars, pivots, pdh, pdl, lookback_days=5, tol_pct=0.08):
    """For each candle: the support level its low touched (price closed above it), the resistance its high touched
    (closed below it), and levels broken by the close. Levels = confirmed swing highs/lows of the last few days
    plus the previous day's high and low."""
    h, l, c = (bars[x].to_numpy(float) for x in "hlc")
    dayno = pd.factorize(bars.day)[0]
    n = len(c)
    tol = tol_pct / 100
    sup, res = np.full(n, np.nan), np.full(n, np.nan)
    brk_up, brk_dn = np.zeros(n, bool), np.zeros(n, bool)
    piv = pivots.sort_values("known_at")
    p_known, p_price = piv.known_at.to_numpy(int), piv.price.to_numpy(float)
    p_day = dayno[piv.i.to_numpy(int)] if len(piv) else np.zeros(0, int)
    j, lv, lvday = 0, [], []
    cur_day = -1
    for t in range(n):
        while j < len(p_known) and p_known[j] <= t:
            lv.append(p_price[j])
            lvday.append(p_day[j])
            j += 1
        if dayno[t] != cur_day:                                   # once a day: drop levels older than the look-back
            cur_day = dayno[t]
            keep = [k for k, d in enumerate(lvday) if d >= cur_day - lookback_days]
            lv, lvday = [lv[k] for k in keep], [lvday[k] for k in keep]
        levels = np.array(lv + [x for x in (pdh[t], pdl[t]) if np.isfinite(x)])
        if not len(levels):
            continue
        below = levels[(levels <= c[t]) & (l[t] <= levels * (1 + tol))]
        above = levels[(levels >= c[t]) & (h[t] >= levels * (1 - tol))]
        if len(below):
            sup[t] = below.max()
        if len(above):
            res[t] = above.min()
        if t > 0 and dayno[t - 1] == dayno[t]:
            brk_up[t] = bool(((levels > c[t - 1]) & (levels < c[t])).any())
            brk_dn[t] = bool(((levels < c[t - 1]) & (levels > c[t])).any())
    return sup, res, brk_up, brk_dn


def supply_demand(bars, atr_v, impulse_atr=1.8, base_max=3, lookback_days=5):
    """Demand zone = the small 'base' candles right before a strong up-move; supply = before a strong down-move.
    Signal on the FIRST later retest that is rejected (bullish candle closing above a demand zone, or the mirror).
    A zone dies when a close goes through it."""
    o, h, l, c = (bars[x].to_numpy(float) for x in "ohlc")
    body, rng, up, lo = candle_parts(o, h, l, c)
    dayno = pd.factorize(bars.day)[0]
    n = len(c)
    buy, sell = np.zeros(n, bool), np.zeros(n, bool)
    zlo, zhi = np.full(n, np.nan), np.full(n, np.nan)
    zones = []                            # [kind(+1 demand/-1 supply), lo, hi, created_at, dayno, alive]
    for t in range(n):
        a = atr_v[t - 1] if t > 0 and np.isfinite(atr_v[t - 1]) else np.nan
        for z in zones:
            if not z[5] or z[3] >= t - 1 or z[4] < dayno[t] - lookback_days:
                continue
            kind, zl, zh = z[0], z[1], z[2]
            if kind == 1:
                if c[t] < zl:
                    z[5] = False
                elif l[t] <= zh:
                    if c[t] > zh and c[t] > o[t] and lo[t] >= 0.3 * rng[t]:
                        buy[t], zlo[t], zhi[t] = True, zl, zh
                    z[5] = False                          # only the first retest counts
            else:
                if c[t] > zh:
                    z[5] = False
                elif h[t] >= zl:
                    if c[t] < zl and c[t] < o[t] and up[t] >= 0.3 * rng[t]:
                        sell[t], zlo[t], zhi[t] = True, zl, zh
                    z[5] = False
        if np.isfinite(a) and rng[t] >= impulse_atr * a and body[t] >= 0.6 * rng[t]:
            b = 0
            while b < base_max and t - b - 1 >= 0 and dayno[t - b - 1] == dayno[t] and rng[t - b - 1] <= 0.6 * a:
                b += 1
            if b:
                bl, bh = l[t - b:t].min(), h[t - b:t].max()
                zones.append([1 if c[t] > o[t] else -1, bl, bh, t, dayno[t], True])
        if len(zones) > 200:
            zones = [z for z in zones if z[5] and z[4] >= dayno[t] - lookback_days]
    return buy, sell, zlo, zhi

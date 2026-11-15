"""Strategy signals A-G and the 0-100 BUY/SELL strength score.
A signal at candle i uses only what is known at i's close. The engine decides entry (that close or the next open)."""
import numpy as np
import pandas as pd

from config import SESSION_START
from indicators import atr, candle_parts, day_prev_high_low
from market_structure import htf_trend_on, sr_levels, structure, supply_demand
from price_action import PATTERN_POINTS, bullish_candle, patterns
from volume_analysis import divergence, high_volume_rejection, relative_volume, volume_class


# ----------------------------------------------------------------------------------------------------------- context
def build_context(bars, htf_bars, s):
    """Everything the strategies and the score need, one row per trading candle."""
    ctx = bars.copy()
    a = atr(bars.h, bars.l, bars.c, 14)
    ctx["atr"] = a
    st = structure(bars, s.pivot_k)
    ctx["trend"], ctx["last_sh"], ctx["last_sl"], ctx["ms_event"] = st["trend"], st["last_sh"], st["last_sl"], st["event"]
    ctx["htf"] = st["trend"] if htf_bars is None else htf_trend_on(bars, s.timeframe, htf_bars, s.htf, s.pivot_k)
    pdh, pdl = day_prev_high_low(bars)
    ctx["pdh"], ctx["pdl"] = pdh, pdl
    sup, res, bu, bd = sr_levels(bars, st["pivots"], pdh, pdl, s.sr_lookback_days, s.sr_tol_pct)
    ctx["sup"], ctx["res"], ctx["brk_up"], ctx["brk_dn"] = sup, res, bu, bd
    rv = relative_volume(bars, s.vol_lookback, s.vol_mode)
    ctx["rv"] = rv
    ctx["hvr"] = high_volume_rejection(bars, rv, s.vol_strong)
    ctx["div"] = divergence(bars, rv)
    ctx["candle"] = bullish_candle(bars)
    sdb, sds, zlo, zhi = supply_demand(bars, a, s.impulse_atr, s.base_max_bars, s.sr_lookback_days)
    ctx["sd_buy"], ctx["sd_sell"], ctx["zone_lo"], ctx["zone_hi"] = sdb, sds, zlo, zhi
    pats = patterns(bars, a)
    return ctx, pats, st["pivots"]


# ----------------------------------------------------------------------------------------------------------- price model (G)
def model_features(C):
    """The NIFTY price model's 12 features (same as the live dashboard) for every day x candle 12..63. C = days x 75 closes."""
    from numpy.lib.stride_tricks import sliding_window_view as swv
    ks = np.arange(12, 64)
    dif = np.diff(C, axis=1, prepend=C[:, :1]) / C * 1e4
    sw = swv(dif, 12, axis=1).std(axis=-1)
    rmax, rmin = np.maximum.accumulate(C, axis=1), np.minimum.accumulate(C, axis=1)
    lag = lambda x: np.concatenate([[0.0], x[:-1]])
    pr, prr = lag((C[:, -1] / C[:, 0] - 1) * 1e4), lag((C.max(1) - C.min(1)) / C[:, 0] * 1e4)
    gap = np.concatenate([[0.0], (C[1:, 0] / C[:-1, -1] - 1) * 1e4])
    er = (C[:, :10].max(1) - C[:, :10].min(1)) / C[:, 0] * 1e4
    sh = (C.shape[0], len(ks))
    X = [(C[:, ks] / C[:, ks - n] - 1) * 1e4 for n in (1, 3, 6, 12)]
    X += [(C[:, ks] / C[:, :1] - 1) * 1e4, (C[:, ks] - rmin[:, ks]) / (rmax[:, ks] - rmin[:, ks] + 1e-9), sw[:, ks - 11],
          np.broadcast_to(er[:, None], sh), np.broadcast_to(5 * ks / 375, sh), np.broadcast_to(pr[:, None], sh),
          np.broadcast_to(prr[:, None], sh), np.broadcast_to(gap[:, None], sh)]
    return np.stack(X, -1), (C[:, -1:] / C[:, ks] - 1) * 1e4


def model_signals(bars):
    """Walk-forward: for each year, train on all earlier years only, then predict that year. Needs 5-min candles.
    A signal = prediction at least as big as the 90th percentile of the training predictions (the live rule)."""
    b5 = bars[(bars.mins >= SESSION_START) & (bars.mins <= SESSION_START + 370)]
    cnt = b5.groupby("day").size()
    full = cnt[cnt == 75].index
    b5 = b5[b5.day.isin(full)]
    if len(full) < 300:
        return pd.DataFrame(columns=["i", "side", "ratio"])
    C = b5.c.to_numpy(float).reshape(-1, 75)
    idx = b5.index.to_numpy().reshape(-1, 75)
    X, Y = model_features(C)
    yr = np.array([d.year for d in full])
    out = []
    for Z in sorted(set(yr))[1:]:
        tr, te = yr < Z, yr == Z
        Xt = X[tr].reshape(-1, 12)
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        A = (Xt - mu) / sd
        y = Y[tr].reshape(-1)
        w = np.linalg.solve(A.T @ A + 0.1 * len(A) * np.eye(12), A.T @ (y - y.mean()))
        thr = np.percentile(np.abs(A @ w), 90)
        P = ((X[te] - mu) / sd) @ w
        r, k = np.where(np.abs(P) >= thr)
        for rr, kk in zip(r, k):
            out.append((int(idx[np.where(te)[0][rr], kk + 12]), 1 if P[rr, kk] > 0 else -1, float(abs(P[rr, kk]) / thr)))
    return pd.DataFrame(out, columns=["i", "side", "ratio"])


# ----------------------------------------------------------------------------------------------------------- signals
def generate_signals(ctx, pats, strategies, model_sig=None):
    """All candidate signals: one row per (candle, strategy). setup = 0-25 points for the quality of the setup."""
    rows = []
    c, o = ctx.c.to_numpy(float), ctx.o.to_numpy(float)
    day, mins = ctx.day.to_numpy(), ctx.mins.to_numpy()
    cand = ctx.candle.to_numpy()
    body, rng, up, lo = candle_parts(ctx.o, ctx.h, ctx.l, ctx.c)
    first_of_day = np.concatenate([[True], day[1:] != day[:-1]])

    def add(mask, side, strat, setup, note):
        for i in np.where(mask)[0]:
            rows.append((int(i), side, strat, float(setup[i] if np.ndim(setup) else setup), note if isinstance(note, str) else note[i]))

    def first_per_day(mask):
        m = pd.Series(mask, index=ctx.index)
        return (m & (m.groupby(ctx.day).cumsum() == 1)).to_numpy()

    if "ORB" in strategies:
        opening = mins < SESSION_START + 15
        g = ctx.assign(oh=np.where(opening, ctx.h, np.nan), ol=np.where(opening, ctx.l, np.nan)).groupby("day")
        orh, orl = g.oh.transform("max").to_numpy(), g.ol.transform("min").to_numpy()
        after = mins >= SESSION_START + 15
        add(first_per_day(after & (c > orh)), 1, "ORB", 15 + 10 * (cand == 1), "close above the 15-min opening range")
        add(first_per_day(after & (c < orl)), -1, "ORB", 15 + 10 * (cand == -1), "close below the 15-min opening range")
    if "PDHL" in strategies:
        pdh, pdl = ctx.pdh.to_numpy(), ctx.pdl.to_numpy()
        pc = np.where(first_of_day, o, np.concatenate([[np.nan], c[:-1]]))
        add(first_per_day((c > pdh) & (pc <= pdh)), 1, "PDHL", 15 + 10 * (cand == 1), "close above previous day high")
        add(first_per_day((c < pdl) & (pc >= pdl)), -1, "PDHL", 15 + 10 * (cand == -1), "close below previous day low")
    if "SR" in strategies:
        bullpat = (pats.engulfing == 1) | (pats.hammer == 1) | (pats.pin_bar == 1)
        bearpat = (pats.engulfing == -1) | (pats.shooting_star == -1) | (pats.pin_bar == -1)
        b = np.isfinite(ctx.sup.to_numpy()) & (c > o) & ((lo >= 0.5 * rng) | bullpat.to_numpy())
        s_ = np.isfinite(ctx.res.to_numpy()) & (c < o) & ((up >= 0.5 * rng) | bearpat.to_numpy())
        add(b, 1, "SR", 15 + 10 * bullpat.to_numpy(), "bullish rejection at support")
        add(s_, -1, "SR", 15 + 10 * bearpat.to_numpy(), "bearish rejection at resistance")
    if "SD" in strategies:
        add(ctx.sd_buy.to_numpy(), 1, "SD", 20, "demand zone retest rejected")
        add(ctx.sd_sell.to_numpy(), -1, "SD", 20, "supply zone retest rejected")
    if "CANDLE" in strategies:
        pts = np.array([PATTERN_POINTS[k] for k in pats.columns])
        P = pats.to_numpy()
        bull_pts = np.where(P == 1, pts, 0)
        bear_pts = np.where(P == -1, pts, 0)
        bsc = bull_pts.max(1) + 5 * ((P == 1).sum(1) >= 2)
        ssc = bear_pts.max(1) + 5 * ((P == -1).sum(1) >= 2)
        names = np.array(pats.columns)
        bnote = np.array([", ".join(names[P[i] == 1]) for i in range(len(P))], dtype=object)
        snote = np.array([", ".join(names[P[i] == -1]) for i in range(len(P))], dtype=object)
        add((bsc > 0) & (bsc > ssc), 1, "CANDLE", np.minimum(bsc, 25), bnote)
        add((ssc > 0) & (ssc > bsc), -1, "CANDLE", np.minimum(ssc, 25), snote)
    if "MS" in strategies:
        ev = ctx.ms_event.to_numpy()
        add(ev == 1, 1, "MS", 22, "break of structure up (BOS)")
        add(ev == 2, 1, "MS", 18, "change of character up (CHoCH)")
        add(ev == -1, -1, "MS", 22, "break of structure down (BOS)")
        add(ev == -2, -1, "MS", 18, "change of character down (CHoCH)")
    if "MODEL" in strategies and model_sig is not None and len(model_sig):
        for i, sd_, r in model_sig[["i", "side", "ratio"]].itertuples(index=False):
            rows.append((int(i), int(sd_), "MODEL", float(min(25, 15 + 5 * (r - 1))), f"price model {r:.1f}x"))
    return pd.DataFrame(rows, columns=["i", "side", "strategy", "setup", "note"])


def score_signals(sig, ctx, s):
    """0-100 score: structure 20, setup 25, volume 20, support/resistance 15, higher timeframe 20."""
    if sig.empty:
        return sig.assign(score=[], cls=[], strict_ok=[])
    i, side = sig.i.to_numpy(), sig.side.to_numpy()
    tr, htf = ctx.trend.to_numpy()[i], ctx.htf.to_numpy()[i]
    rv = ctx.rv.to_numpy()[i]
    vcls = volume_class(rv, s.vol_strong, s.vol_moderate)
    s_struct = np.where(tr == side, 20, np.where(tr == 0, 10, 0))
    s_setup = np.minimum(sig.setup.to_numpy(), 25)
    s_vol = np.where(vcls == 2, 20, np.where(vcls == 1, 12, 0))
    sup_ok = np.isfinite(ctx.sup.to_numpy()[i]) | ctx.brk_up.to_numpy()[i]
    res_ok = np.isfinite(ctx.res.to_numpy()[i]) | ctx.brk_dn.to_numpy()[i]
    s_sr = np.where(np.where(side == 1, sup_ok, res_ok), 15, 0)
    s_htf = np.where(htf == side, 20, np.where(htf == 0, 10, 0))
    score = s_struct + s_setup + s_vol + s_sr + s_htf
    out = sig.copy()
    out["sc_structure"], out["sc_setup"], out["sc_volume"], out["sc_sr"], out["sc_htf"] = s_struct, s_setup, s_vol, s_sr, s_htf
    out["score"] = score.round().astype(int)
    out["cls"] = np.where(score >= 80, "Very Strong", np.where(score >= 65, "Strong", np.where(score >= 50, "Moderate", "Avoid")))
    out["rv"], out["vol_class"] = rv, vcls
    out["strict_ok"] = (tr == side) & (ctx.candle.to_numpy()[i] == side) & (vcls >= 1)
    return out

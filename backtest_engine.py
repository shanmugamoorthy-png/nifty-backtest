"""The backtest engine. One position at a time, at most N trades a day, everything closed by 15:15.
Exits are checked on 1-minute candles. If a stop and a target fall inside the same minute, the STOP is assumed first."""
import numpy as np
import pandas as pd

from risk_management import order_costs, stop_level, target_level

NS_MIN = 60_000_000_000


def _best_signals(sc, s, n):
    """For every candle, the row of the best allowed signal (highest score, then setup), or -1. Plain numpy: low memory."""
    best = np.full(n, -1, dtype=np.int64)
    if sc.empty:
        return best
    m = sc.strategy.isin(s.strategies).to_numpy() & (sc.score.to_numpy() >= s.min_score)
    if s.strict:
        m &= sc.strict_ok.to_numpy(bool)
    idx = np.where(m)[0]
    if not len(idx):
        return best
    order = idx[np.lexsort((-sc.setup.to_numpy(float)[idx], -sc.score.to_numpy(float)[idx]))]
    ii = sc.i.to_numpy(np.int64)[order]
    _, first = np.unique(ii, return_index=True)
    best[ii[first]] = order[first]
    return best


def _simulate(side, entry_ns, entry_px, sl, tp, s, M, bars, tf):
    """Walk 1-minute candles after the entry. Returns fills, exit time (ns), exit reason, highest favourable move."""
    t0 = np.searchsorted(M["t"], entry_ns, side="left")
    day_end = M["day_end"][t0] if t0 < len(M["t"]) else t0
    qty_left, fills = 1.0, []
    R = abs(entry_px - sl) if sl is not None else None
    be_done = trail_on = part_done = False
    slip = s.slippage_pts
    reason, exit_ns = "EOD", None
    bt, bl, bh = bars["t"], bars["l"], bars["h"]
    j = t0
    while j < day_end:
        tm, mins = M["t"][j], M["mins"][j]
        o, h, l = M["o"][j], M["h"][j], M["l"][j]
        if mins >= s.square_off:
            fills.append((o - side * slip, qty_left)); reason, exit_ns = "EOD 15:15", tm; break
        if s.max_hold_bars and (tm - entry_ns) >= s.max_hold_bars * tf * NS_MIN:
            fills.append((o - side * slip, qty_left)); reason, exit_ns = "TIME", tm; break
        if trail_on and sl is not None and (mins - 555) % tf == 0:
            k = np.searchsorted(bt, tm, side="left") - 1          # the trading candle that just closed
            if k >= 0:
                lvl = bl[k] - s.sl_buffer_pts if side == 1 else bh[k] + s.sl_buffer_pts
                sl = max(sl, lvl) if side == 1 else min(sl, lvl)
        lab = "TRAIL" if trail_on else ("BREAKEVEN" if be_done else "SL")
        if sl is not None and (o - sl) * side <= 0:
            fills.append((o - side * slip, qty_left)); reason, exit_ns = lab + " (gap)", tm; break
        if tp is not None and (o - tp) * side >= 0:
            fills.append((o, qty_left)); reason, exit_ns = "TARGET", tm; break
        hit_sl = sl is not None and ((l <= sl) if side == 1 else (h >= sl))
        hit_tp = tp is not None and ((h >= tp) if side == 1 else (l <= tp))
        if hit_sl:
            fills.append((sl - side * slip, qty_left)); reason, exit_ns = lab, tm; break
        if hit_tp:
            fills.append((tp, qty_left)); reason, exit_ns = "TARGET", tm; break
        best = h if side == 1 else l
        if R:
            if s.partial and not part_done and (best - entry_px) * side >= s.partial_at_r * R:
                fills.append((entry_px + side * s.partial_at_r * R, s.partial_frac))
                qty_left -= s.partial_frac
                part_done = True
            if s.breakeven and not be_done and (best - entry_px) * side >= s.be_at_r * R:
                sl = entry_px if side == 1 else entry_px
                be_done = True
            if s.trailing and not trail_on and (best - entry_px) * side >= s.trail_at_r * R:
                trail_on = True
        j += 1
    if exit_ns is None:                                            # data ended before 15:15
        k = max(day_end - 1, t0)
        fills.append((M["c"][k] - side * slip, qty_left))
        exit_ns, reason = M["t"][k] + NS_MIN, "EOD (data end)"
    return fills, exit_ns, reason


def _minute_arrays(minutes):
    t = minutes.dt.to_numpy("datetime64[ns]").astype("int64")
    day = (t // (86400 * 1_000_000_000))
    ends = np.r_[np.where(np.diff(day) != 0)[0] + 1, len(t)]
    day_end = np.repeat(ends, np.diff(np.r_[0, ends]))
    return {"t": t, "mins": (minutes.dt.dt.hour * 60 + minutes.dt.dt.minute).to_numpy(), "o": minutes.o.to_numpy(float),
            "h": minutes.h.to_numpy(float), "l": minutes.l.to_numpy(float), "c": minutes.c.to_numpy(float), "day_end": day_end}


def run_backtest(ctx, scored, minutes, s, date_from=None, date_to=None, M=None):
    """Returns a DataFrame of trades. `minutes` = 1-min candles (None = use the trading candles themselves)."""
    tf = s.timeframe
    if M is None:
        M = _minute_arrays(minutes if minutes is not None and len(minutes) else ctx[["dt", "o", "h", "l", "c"]])
    bars = {"t": ctx.dt.to_numpy("datetime64[ns]").astype("int64"), "l": ctx.l.to_numpy(float), "h": ctx.h.to_numpy(float)}
    best = _best_signals(scored, s, len(ctx))
    if not scored.empty:
        S_side, S_strat, S_score, S_cls = scored.side.to_numpy(), scored.strategy.to_numpy(), scored.score.to_numpy(), scored.cls.to_numpy()
        S_note, S_rv, S_vc = scored.note.to_numpy(), scored.rv.to_numpy(float), scored.vol_class.to_numpy()
    qty = s.lot_size * s.lots
    dts, mins_, days, c_, o_ = ctx.dt, ctx.mins.to_numpy(), ctx.day.to_numpy(), ctx.c.to_numpy(float), ctx.o.to_numpy(float)
    bar_t = bars["t"]
    sel = np.ones(len(ctx), bool)
    if date_from is not None:
        sel &= ctx.day.to_numpy() >= np.datetime64(pd.Timestamp(date_from))
    if date_to is not None:
        sel &= ctx.day.to_numpy() <= np.datetime64(pd.Timestamp(date_to))
    trades = []
    n = len(ctx)
    i = 0
    count_day, cur_day = 0, None
    while i < n:
        if not sel[i]:
            i += 1
            continue
        if days[i] != cur_day:
            cur_day, count_day = days[i], 0
        r = best[i]
        close_min = mins_[i] + tf
        if r < 0 or count_day >= s.max_trades_day or close_min < s.first_entry or close_min > s.last_entry:
            i += 1
            continue
        side = int(S_side[r])
        if s.entry_mode == "next_open":
            if i + 1 >= n or days[i + 1] != days[i]:
                i += 1
                continue
            e_ns, e_px = bar_t[i + 1], o_[i + 1]
        else:
            e_ns, e_px = bar_t[i] + tf * NS_MIN, c_[i]
        e_px = e_px + side * s.slippage_pts
        sl = stop_level(side, e_px, i, ctx, s)
        tp = target_level(side, e_px, sl, s)
        fills, x_ns, reason = _simulate(side, e_ns, e_px, sl, tp, s, M, bars, tf)
        q = sum(f[1] for f in fills)
        x_px = sum(p * f for p, f in fills) / q
        pts = side * (x_px - e_px)
        cost = order_costs([(e_px, qty, side == 1)] + [(p, qty * f, side == -1) for p, f in fills], s)
        risk = abs(e_px - sl) if sl is not None else np.nan
        trades.append({
            "day": pd.Timestamp(days[i]).date(), "strategy": S_strat[r], "side": "BUY" if side == 1 else "SELL",
            "score": int(S_score[r]), "class": S_cls[r], "setup": S_note[r], "rv": float(S_rv[r]) if np.isfinite(S_rv[r]) else np.nan,
            "vol_class": int(S_vc[r]), "signal_time": pd.Timestamp(bar_t[i] + tf * NS_MIN), "entry_time": pd.Timestamp(e_ns),
            "entry": round(e_px, 2), "stop": round(sl, 2) if sl is not None else np.nan, "target": round(tp, 2) if tp is not None else np.nan,
            "exit_time": pd.Timestamp(x_ns), "exit": round(x_px, 2), "reason": reason, "points": round(pts, 2),
            "risk_pts": round(risk, 2), "r_multiple": round(pts / risk, 2) if risk and np.isfinite(risk) else np.nan,
            "gross": round(pts * qty, 2), "costs": round(cost, 2), "net": round(pts * qty - cost, 2),
            "minutes_held": int((x_ns - e_ns) // NS_MIN)})
        count_day += 1
        nxt = np.searchsorted(bar_t + tf * NS_MIN, x_ns, side="right")        # next candle that closes after the exit
        i = max(i + 1, int(nxt))
    return pd.DataFrame(trades)

"""Stop loss, target and trading costs."""
import numpy as np


def stop_level(side, entry, i, ctx, s):
    """Initial stop. 'signal' = beyond the signal candle, 'swing' = beyond the last confirmed swing point,
    'fixed' = fixed points, 'none' = no stop (exit by target / time / 15:15 only). Kept between min and max points."""
    if s.sl_method == "none":
        return None
    lo, hi = float(ctx.l.iat[i]), float(ctx.h.iat[i])
    if s.sl_method == "fixed":
        ref = entry - side * s.sl_fixed_pts
    elif s.sl_method == "swing":
        sw = ctx.last_sl.iat[i] if side == 1 else ctx.last_sh.iat[i]
        ok = np.isfinite(sw) and ((side == 1 and sw < entry) or (side == -1 and sw > entry))
        ref = (sw - side * s.sl_buffer_pts) if ok else ((lo if side == 1 else hi) - side * s.sl_buffer_pts)
    else:
        ref = (lo if side == 1 else hi) - side * s.sl_buffer_pts
    risk = abs(entry - ref)
    risk = min(max(risk, s.sl_min_pts), s.sl_max_pts)
    return entry - side * risk


def target_level(side, entry, sl, s):
    if s.rr <= 0 or sl is None:
        return None
    return entry + side * s.rr * abs(entry - sl)


def order_costs(fills, s):
    """Rupee costs of a list of fills (price, quantity, is_buy): brokerage, STT, exchange, SEBI, stamp duty, GST."""
    total = 0.0
    for px, q, is_buy in fills:
        v = px * q
        brok = min(s.brokerage_flat, s.brokerage_pct / 100 * v)
        exch = s.exch_pct / 100 * v
        sebi = s.sebi_per_cr * v / 1e7
        stt = 0.0 if is_buy else s.stt_sell_pct / 100 * v
        stamp = s.stamp_buy_pct / 100 * v if is_buy else 0.0
        gst = s.gst_pct / 100 * (brok + exch + sebi)
        total += brok + exch + sebi + stt + stamp + gst
    return total

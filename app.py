"""NIFTY intraday price-action backtester (research only, no live trading, no orders).
Run:  streamlit run app.py      Data: Dhan (DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN) or your own CSV."""
import hashlib
import os
import threading
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st

import charts
from backtest_engine import _minute_arrays, run_backtest
from config import ALL_STRATEGIES, DB_PATH, DEFAULTS, Settings
from data_loader import build_bars, clean, download_job, load_minutes, load_volume5, parse_csv, stored_summary
from performance import (dow_table, equity_curve, group_table, monthly_table, split_in_out, summarize, walk_forward)
from strategies import build_context, generate_signals, model_signals, score_signals
from volume_analysis import VOL_NAMES

st.set_page_config(page_title="NIFTY backtester", page_icon="📊", layout="wide")


# ----------------------------------------------------------------------------------------------------------- data (cached)
@st.cache_resource(show_spinner=False)
def dl_state():
    return {"running": False, "msg": "", "done": 0, "total": 0, "errors": []}


def _db_stamp():
    try:
        return f"{os.path.getmtime(DB_PATH):.0f}"
    except OSError:
        return "none"


@st.cache_resource(max_entries=2, show_spinner="Loading saved data...")
def data_from_db(stamp):
    m, rep = clean(load_minutes())
    return m, load_volume5(), rep


@st.cache_resource(max_entries=2, show_spinner="Reading your CSV...")
def data_from_csv(digest, raw):
    import io
    d, rep = clean(parse_csv(io.BytesIO(raw)))
    if rep["interval_min"] <= 1:
        m = d
    else:
        m = None                                                   # no 1-min data: exits are checked on the CSV candles
    vol = d[["dt", "v"]] if d.v.notna().any() else None
    return (m if m is not None else d), vol, rep


@st.cache_resource(max_entries=3, show_spinner="Finding patterns, structure, levels and zones (first time takes a minute)...")
def context(src_key, tf, htf, pivot_k, sr_days, sr_tol, impulse, base_max, vol_mode, vol_lb, vol_strong, _minutes, _vol):
    s = Settings(timeframe=tf, htf=htf, pivot_k=pivot_k, sr_lookback_days=sr_days, sr_tol_pct=sr_tol, impulse_atr=impulse,
                 base_max_bars=base_max, vol_mode=vol_mode, vol_lookback=vol_lb, vol_strong=vol_strong)
    bars = build_bars(_minutes, _vol, tf)
    hb = build_bars(_minutes, _vol, htf) if htf != tf else None
    ctx, pats, piv = build_context(bars, hb, s)
    ms = model_signals(bars) if tf == 5 else None
    sig = generate_signals(ctx, pats, tuple(ALL_STRATEGIES), ms)
    return ctx, sig, piv


@st.cache_resource(max_entries=2, show_spinner=False)
def minute_arrays(src_key, _minutes):
    return _minute_arrays(_minutes)


# ----------------------------------------------------------------------------------------------------------- sidebar
st.title("📊 NIFTY price-action backtester")
st.caption("Backtest only: no live signals, no orders. Your live NIFTY paper log is untouched. Results are research, not advice.")

with st.sidebar:
    st.header("Data")
    src = st.radio("Source", ["Dhan (saved)", "Upload CSV"], horizontal=True)
    up = None
    if src == "Dhan (saved)":
        S_ = dl_state()
        st.caption(stored_summary())
        yrs = st.select_slider("Years to download", [1, 2, 3, 4, 5], value=5)
        if st.button("⬇️ Download / update data", disabled=S_["running"], use_container_width=True):
            threading.Thread(target=download_job, args=(S_, float(yrs)), daemon=True).start()
            S_.update(running=True, msg="starting...")
        if S_["running"] or S_["msg"]:
            pct = S_["done"] / S_["total"] if S_["total"] else 0
            st.progress(min(pct, 1.0), text=f"{'Running' if S_['running'] else 'Idle'}: {S_['msg']}")
            for e in S_["errors"][:5]:
                st.caption("note: " + e)
            if S_["running"]:
                st.button("🔄 Refresh progress", use_container_width=True)
    else:
        up = st.file_uploader("CSV: datetime, open, high, low, close, volume", type=["csv"])

    st.header("Strategy")
    picked = st.multiselect("Strategies", list(ALL_STRATEGIES), default=list(DEFAULTS.strategies), format_func=lambda k: ALL_STRATEGIES[k])
    tf = st.radio("Trading timeframe", [5, 15], horizontal=True, format_func=lambda x: f"{x} min",
                  help="Built from 1-minute candles. Confirmation timeframe: 15 min (for 15-min trading: 60 min).")
    entry_mode = st.radio("Entry", ["close", "next_open"], horizontal=True, format_func=lambda x: "signal close" if x == "close" else "next open")
    min_score = st.slider("Minimum score", 0, 100, DEFAULTS.min_score, 5, help="80+ very strong, 65+ strong, 50+ moderate, below 50 = avoid")
    strict = st.toggle("Strict entry rules", value=False, help="Also require: structure agrees, a bullish/bearish candle and volume at least moderate")
    max_tr = st.number_input("Max trades per day", 1, 10, DEFAULTS.max_trades_day)

    with st.expander("Stop loss and target"):
        sl_method = st.selectbox("Stop loss", ["signal", "swing", "fixed", "none"],
                                 format_func={"signal": "Signal candle high/low", "swing": "Previous swing high/low",
                                              "fixed": "Fixed points", "none": "No stop"}.get)
        sl_fixed = st.number_input("Fixed stop (points)", 5.0, 300.0, DEFAULTS.sl_fixed_pts, 5.0)
        sl_buf = st.number_input("Stop buffer (points)", 0.0, 50.0, DEFAULTS.sl_buffer_pts, 1.0)
        c1, c2 = st.columns(2)
        sl_min = c1.number_input("Min stop", 1.0, 200.0, DEFAULTS.sl_min_pts, 1.0)
        sl_max = c2.number_input("Max stop", 5.0, 500.0, DEFAULTS.sl_max_pts, 5.0)
        rr = st.select_slider("Risk : reward", [0.0, 1.0, 1.5, 2.0, 3.0], value=DEFAULTS.rr, format_func=lambda x: "no target" if x == 0 else f"1:{x:g}")
        be = st.toggle("Break-even stop at 1R", value=DEFAULTS.breakeven)
        trail = st.toggle("Trailing stop (behind each candle, after 1R)", value=DEFAULTS.trailing)
        part = st.toggle("Book 50% at 1R", value=DEFAULTS.partial)
        hold = st.number_input("Time exit after N candles (0 = off)", 0, 75, 0)
        sq = st.select_slider("Square off", ["15:00", "15:10", "15:15", "15:20"], value="15:15")

    with st.expander("Volume"):
        vol_mode = st.radio("Average volume", ["candles", "time"], format_func=lambda x: "previous 20 candles" if x == "candles" else "same time, previous 20 days",
                            help="NIFTY volume = traded value of the 50 index stocks. 'Same time' is fairer: 09:15 is always busy.")
        vs = st.number_input("Strong breakout ≥ x average", 1.0, 5.0, DEFAULTS.vol_strong, 0.1)
        vm = st.number_input("Moderate breakout ≥ x average", 1.0, 5.0, DEFAULTS.vol_moderate, 0.1)

    with st.expander("Structure and levels"):
        piv_k = st.slider("Swing point strength (candles each side)", 2, 6, DEFAULTS.pivot_k)
        sr_days = st.slider("Support/resistance look-back (days)", 1, 20, DEFAULTS.sr_lookback_days)
        sr_tol = st.number_input("At a level within (%)", 0.01, 0.5, DEFAULTS.sr_tol_pct, 0.01)
        imp = st.number_input("Impulsive candle ≥ x ATR", 1.0, 4.0, DEFAULTS.impulse_atr, 0.1)

    with st.expander("Costs (NIFTY futures)"):
        lot = st.number_input("Lot size", 1, 500, DEFAULTS.lot_size)
        lots = st.number_input("Lots", 1, 100, DEFAULTS.lots)
        slip = st.number_input("Slippage per fill (points)", 0.0, 20.0, DEFAULTS.slippage_pts, 0.5)
        brk = st.number_input("Brokerage per order (₹, max)", 0.0, 100.0, DEFAULTS.brokerage_flat, 1.0)
        stt = st.number_input("STT on sell (%)", 0.0, 0.5, DEFAULTS.stt_sell_pct, 0.005, format="%.4f")
        exch = st.number_input("Exchange charges (%)", 0.0, 0.1, DEFAULTS.exch_pct, 0.0001, format="%.5f")
        stamp = st.number_input("Stamp duty on buy (%)", 0.0, 0.1, DEFAULTS.stamp_buy_pct, 0.001, format="%.4f")
        gst = st.number_input("GST (%)", 0.0, 30.0, DEFAULTS.gst_pct, 1.0)

# ----------------------------------------------------------------------------------------------------------- load data
try:
    if src == "Dhan (saved)":
        minutes, vol, rep = data_from_db(_db_stamp())
        src_key = "db" + _db_stamp()
    elif up is not None:
        raw = up.getvalue()
        dg = hashlib.md5(raw).hexdigest()
        minutes, vol, rep = data_from_csv(dg, raw)
        src_key = "csv" + dg
    else:
        minutes = None
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()
if minutes is not None and rep["interval_min"] > tf:
    st.error(f"Your CSV has {rep['interval_min']}-min candles; {tf}-min candles cannot be built from them. Pick a larger timeframe.")
    st.stop()
if minutes is None or minutes.dt.dt.date.nunique() < 60:
    st.info("Not enough data yet. In the sidebar press **Download / update data** (needs DHAN_CLIENT_ID and "
            "DHAN_ACCESS_TOKEN; about 20-40 minutes for 5 years) or upload a CSV. At least about 60 trading days are needed.")
    st.stop()

d_min, d_max = minutes.dt.min().date(), minutes.dt.max().date()
c1, c2, c3 = st.columns(3)
dfrom = c1.date_input("From", d_min, min_value=d_min, max_value=d_max)
dto = c2.date_input("To", d_max, min_value=d_min, max_value=d_max)
cut_default = (pd.Timestamp(d_min) + (pd.Timestamp(d_max) - pd.Timestamp(d_min)) * 0.7).date()
cutoff = c3.date_input("Out-of-sample starts", cut_default, min_value=d_min, max_value=d_max,
                       help="Tune settings on data before this date only; judge them on the data after it.")

htf = 15 if tf == 5 else 60
sqm = {"15:00": 900, "15:10": 910, "15:15": 915, "15:20": 920}[sq]
S = Settings(timeframe=tf, htf=htf, strategies=tuple(picked), entry_mode=entry_mode, min_score=int(min_score), strict=strict,
             max_trades_day=int(max_tr), last_entry=min(DEFAULTS.last_entry, sqm - 15), square_off=sqm, sl_method=sl_method,
             sl_fixed_pts=sl_fixed, sl_buffer_pts=sl_buf, sl_min_pts=sl_min, sl_max_pts=max(sl_max, sl_min), rr=float(rr),
             breakeven=be, trailing=trail, partial=part, max_hold_bars=int(hold), vol_mode=vol_mode, vol_strong=vs,
             vol_moderate=min(vm, vs), pivot_k=piv_k, sr_lookback_days=sr_days, sr_tol_pct=sr_tol, impulse_atr=imp,
             lot_size=int(lot), lots=int(lots), slippage_pts=slip, brokerage_flat=brk, stt_sell_pct=stt, exch_pct=exch,
             stamp_buy_pct=stamp, gst_pct=gst)
if not picked:
    st.warning("Pick at least one strategy in the sidebar.")
    st.stop()

ctx, sig, piv = context(src_key, tf, htf, piv_k, sr_days, sr_tol, imp, DEFAULTS.base_max_bars, vol_mode, DEFAULTS.vol_lookback,
                        vs, minutes, vol)
M = minute_arrays(src_key, minutes)
scored = score_signals(sig, ctx, S)
if "MODEL" in picked and tf != 5:
    st.caption("The NIFTY price model (G) works on 5-min candles only; it is skipped on 15-min.")
if ctx.v.isna().all():
    st.caption("No volume in this data: volume points are 0 and strict mode will find no trades.")

trades = run_backtest(ctx, scored, minutes, S, dfrom, dto, M=M)

# ----------------------------------------------------------------------------------------------------------- results
tabs = st.tabs(["Overview", "Monthly", "Trades", "Chart", "Strategies", "Volume", "Optimize", "Data"])

with tabs[0]:
    sm = summarize(trades)
    if not len(trades):
        st.info("No trades with these settings. Lower the minimum score or turn off strict rules.")
    else:
        k = st.columns(4)
        k[0].metric("Net profit ₹", f"{sm['Net profit ₹']:,.0f}")
        k[1].metric("Trades", f"{sm['Total trades']:,}", f"{sm['Win rate %']}% win")
        k[2].metric("Profit factor", f"{sm['Profit factor']}")
        k[3].metric("Max drawdown ₹", f"{sm['Max drawdown ₹']:,.0f}")
        eq = equity_curve(trades)
        st.plotly_chart(charts.equity_fig(eq), use_container_width=True)
        st.plotly_chart(charts.drawdown_fig(eq), use_container_width=True)
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**All numbers**")
            st.dataframe(pd.DataFrame({"value": [str(v) for v in sm.values()]}, index=list(sm.keys())), use_container_width=True)
        with cb:
            st.markdown("**In-sample vs out-of-sample**")
            sp = split_in_out(trades, cutoff)
            if sp is not None:
                st.dataframe(sp.astype(str), use_container_width=True)
            st.caption("Only the out-of-sample column is a fair test, and only if you did not tune settings while looking at it. "
                       f"Average cost per round trip: ₹{trades.costs.mean():,.0f} "
                       f"(≈ {trades.costs.mean() / (S.lot_size * S.lots):.1f} points), plus {2 * S.slippage_pts:g} points slippage.")

with tabs[1]:
    if len(trades):
        mt = monthly_table(trades)
        st.plotly_chart(charts.monthly_fig(mt), use_container_width=True)
        st.dataframe(mt.style.format("{:,.0f}", na_rep=""), use_container_width=True)
        dw = dow_table(trades)
        st.plotly_chart(charts.bar_fig(dw, "weekday", "net ₹", "Net ₹ by day of week"), use_container_width=True)

with tabs[2]:
    if len(trades):
        show = trades.copy()
        show["volume"] = show.vol_class.map(VOL_NAMES)
        cols = ["day", "strategy", "side", "score", "class", "setup", "volume", "entry_time", "entry", "stop", "target",
                "exit_time", "exit", "reason", "points", "r_multiple", "costs", "net"]
        st.dataframe(show[cols], use_container_width=True, hide_index=True,
                     column_config={"entry_time": st.column_config.DatetimeColumn(format="HH:mm"),
                                    "exit_time": st.column_config.DatetimeColumn(format="HH:mm")})
        st.download_button("⬇️ Download trade history (CSV)", trades.to_csv(index=False), "nifty_backtest_trades.csv", "text/csv")

with tabs[3]:
    tdays = sorted(set(trades.day), reverse=True) if len(trades) else []
    alld = sorted(set(ctx.day.dt.date), reverse=True)
    only = st.toggle("Only days with trades", value=bool(tdays))
    opts = tdays if only and tdays else alld
    dsel = st.selectbox("Day", opts, format_func=lambda d: f"{d:%d %b %Y (%a)}")
    cd = ctx[ctx.day.dt.date == dsel]
    td = trades[trades.day == dsel] if len(trades) else trades
    pv = piv.copy()
    pv["dt"] = ctx.dt.to_numpy()[pv.i.to_numpy(int)] if len(pv) else []
    pv = pv[pd.to_datetime(pv.dt).dt.date == dsel] if len(pv) else pv
    st.plotly_chart(charts.day_fig(cd, td, pv), use_container_width=True)
    if len(td):
        st.dataframe(td[["strategy", "side", "score", "setup", "entry_time", "entry", "stop", "target", "exit_time", "exit", "reason",
                         "points", "net"]], use_container_width=True, hide_index=True,
                     column_config={"entry_time": st.column_config.DatetimeColumn(format="HH:mm"),
                                    "exit_time": st.column_config.DatetimeColumn(format="HH:mm")})
    st.caption("▲/▼ entry, ■ exit (green = profit), dashed red = stop, dashed green = target, dotted = previous day high/low. "
               "HH/HL/LH/LL labels sit on the swing candle but are only known a few candles later.")

with tabs[4]:
    if len(trades):
        gt = group_table(trades, "strategy")
        gt["strategy"] = gt.strategy.map(lambda k: ALL_STRATEGIES.get(k, k))
        st.plotly_chart(charts.bar_fig(gt, "strategy", "net ₹", "Net ₹ by strategy"), use_container_width=True)
        st.dataframe(gt, use_container_width=True, hide_index=True)
        st.markdown("**BUY vs SELL**")
        bs = group_table(trades, "side")
        st.dataframe(bs, use_container_width=True, hide_index=True)
        st.markdown("**By score class**")
        st.dataframe(group_table(trades, "class"), use_container_width=True, hide_index=True)
        st.markdown("**By exit reason**")
        st.dataframe(group_table(trades, "reason"), use_container_width=True, hide_index=True)

with tabs[5]:
    if len(trades) and trades.rv.notna().any():
        st.plotly_chart(charts.vol_fig(trades), use_container_width=True)
        st.plotly_chart(charts.rv_hist(trades), use_container_width=True)
        st.caption(f"Strong = volume ≥ {S.vol_strong}x average, moderate ≥ {S.vol_moderate}x. If volume confirmation works, "
                   "'strong' should earn clearly more than 'weak'.")
    else:
        st.info("No volume data for these trades.")

with tabs[6]:
    st.markdown("**Walk-forward optimisation**: 36 combinations (risk:reward × stop type × minimum score). For each year, the "
                "combination that did best in the EARLIER years is picked and judged on that year. This is the honest number.")
    if st.button("▶️ Run walk-forward (36 backtests)"):
        bar = st.progress(0.0, text="running...")
        run = lambda st_: run_backtest(ctx, score_signals(sig, ctx, st_), minutes, st_, dfrom, dto, M=M)
        tab, oos, grid = walk_forward(run, S, progress=lambda i, n: bar.progress(i / n, text=f"backtest {i + 1} of {n}"))
        bar.empty()
        st.session_state["wf"] = (tab, oos, grid)
    if "wf" in st.session_state:
        tab, oos, grid = st.session_state["wf"]
        st.dataframe(tab, use_container_width=True, hide_index=True)
        if len(oos):
            so = summarize(oos)
            st.success(f"Stitched out-of-sample result: {so['Total trades']} trades, net ₹{so['Net profit ₹']:,.0f}, "
                       f"profit factor {so['Profit factor']}, win {so['Win rate %']}%")
        slm = st.radio("Sensitivity map for stop type", ["signal", "swing", "fixed"], horizontal=True)
        st.plotly_chart(charts.sensitivity_fig(grid, slm), use_container_width=True)
        st.caption("A good setting sits in a broad green area, not on a lone green square. The map uses the whole period, so it "
                   "is NOT a performance number.")

with tabs[7]:
    st.markdown("**Data quality**")
    st.dataframe(pd.DataFrame({"value": [str(v) for v in rep.values()]}, index=list(rep.keys())), use_container_width=True)
    st.caption(f"Trading candles: {len(ctx):,} of {tf} min. Signals found (all strategies, before filters): {len(sig):,}. "
               "NIFTY volume = 5-min traded value (₹ crore) of the NIFTY 50 stocks added together.")

"""Plotly charts for the dashboard."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GREEN, RED, BLUE, GREY = "#16a34a", "#dc2626", "#2563eb", "#9ca3af"
LAYOUT = dict(margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", hovermode="x unified",
              legend=dict(orientation="h", y=1.08, x=0))


def _base(fig, height=360, title=None):
    """Same look everywhere. Charts are locked (no drag/zoom) so scrolling on a phone does not move them."""
    fig.update_layout(height=height, title=title, dragmode=False, **LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", fixedrange=True)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", fixedrange=True)
    return fig


PLOT_CONFIG = {"displayModeBar": False, "scrollZoom": False, "doubleClick": False}


def equity_fig(eq):
    fig = go.Figure(go.Scatter(x=eq.time, y=eq.equity, mode="lines", line=dict(color=BLUE, width=2), name="Equity ₹"))
    fig.add_hline(y=0, line_color=GREY, line_width=1)
    return _base(fig, 340, "Equity curve (₹, after costs)")


def drawdown_fig(eq):
    fig = go.Figure(go.Scatter(x=eq.time, y=eq.drawdown, mode="lines", fill="tozeroy", line=dict(color=RED, width=1), name="Drawdown ₹"))
    return _base(fig, 240, "Drawdown (₹ below the best equity so far)")


def monthly_fig(tab):
    t = tab.drop(columns="Total", errors="ignore")
    z = t.to_numpy(float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
    fig = go.Figure(go.Heatmap(z=z, x=list(t.columns), y=[str(y) for y in t.index], zmid=0, zmin=-lim, zmax=lim,
                               colorscale=[[0, RED], [0.5, "#ffffff"], [1, GREEN]],
                               text=[[("" if not np.isfinite(v) else f"{v / 1000:,.0f}k") for v in row] for row in z],
                               texttemplate="%{text}", hovertemplate="%{y} %{x}: ₹%{z:,.0f}<extra></extra>", showscale=False))
    fig.update_yaxes(autorange="reversed", type="category")
    return _base(fig, 60 + 34 * len(t), "Monthly net profit (₹)")


def bar_fig(tab, x, y, title, color_by_sign=True, height=300):
    col = [GREEN if v > 0 else RED for v in tab[y]] if color_by_sign else BLUE
    fig = go.Figure(go.Bar(x=tab[x].astype(str), y=tab[y], marker_color=col, text=[f"{v:,.0f}" for v in tab[y]], textposition="outside"))
    return _base(fig, height, title)


def day_fig(ctx_day, trades_day, pivots_day=None, show_volume=True):
    """Candles of one day with every trade: entry ▲/▼, exit ■, stop and target lines, previous day high/low."""
    rows = 2 if show_volume and ctx_day.v.notna().any() else 1
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=[0.75, 0.25] if rows == 2 else [1], vertical_spacing=0.03)
    fig.add_trace(go.Candlestick(x=ctx_day.dt, open=ctx_day.o, high=ctx_day.h, low=ctx_day.l, close=ctx_day.c, name="NIFTY",
                                 increasing_line_color=GREEN, increasing_fillcolor=GREEN, decreasing_line_color=RED,
                                 decreasing_fillcolor=RED, showlegend=False), row=1, col=1)
    x0, x1 = ctx_day.dt.min(), ctx_day.dt.max()
    for lvl, nm in ((ctx_day.pdh.iloc[0], "PDH"), (ctx_day.pdl.iloc[0], "PDL")):
        if np.isfinite(lvl):
            fig.add_trace(go.Scatter(x=[x0, x1], y=[lvl, lvl], mode="lines+text", line=dict(color=GREY, dash="dot", width=1),
                                     text=[nm, ""], textposition="top right", showlegend=False, hoverinfo="skip"), row=1, col=1)
    if pivots_day is not None and len(pivots_day):
        fig.add_trace(go.Scatter(x=pivots_day.dt, y=pivots_day.price, mode="text", text=pivots_day.label,
                                 textposition=["top center" if lb in ("HH", "LH", "SH") else "bottom center" for lb in pivots_day.label],
                                 textfont=dict(size=10, color="#6b7280"), showlegend=False, hoverinfo="skip"), row=1, col=1)
    for t in trades_day.itertuples():
        up = t.side == "BUY"
        fig.add_trace(go.Scatter(x=[t.entry_time], y=[t.entry], mode="markers", name=f"{t.side} {t.strategy}",
                                 marker=dict(symbol="triangle-up" if up else "triangle-down", size=14, color=BLUE),
                                 hovertext=f"{t.side} {t.strategy} score {t.score} at {t.entry:,.1f}<br>{t.setup}", hoverinfo="text",
                                 showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=[t.exit_time], y=[t.exit], mode="markers", marker=dict(symbol="square", size=10,
                                 color=GREEN if t.net > 0 else RED), hovertext=f"exit {t.exit:,.1f} ({t.reason}) {t.points:+.1f} pts, ₹{t.net:,.0f}",
                                 hoverinfo="text", showlegend=False), row=1, col=1)
        for lvl, colr in ((t.stop, RED), (t.target, GREEN)):
            if np.isfinite(lvl):
                fig.add_trace(go.Scatter(x=[t.entry_time, t.exit_time], y=[lvl, lvl], mode="lines", line=dict(color=colr, dash="dash", width=1),
                                         showlegend=False, hoverinfo="skip"), row=1, col=1)
    if rows == 2:
        fig.add_trace(go.Bar(x=ctx_day.dt, y=ctx_day.v, marker_color=[GREEN if c >= o else RED for o, c in zip(ctx_day.o, ctx_day.c)],
                             opacity=0.6, name="NIFTY 50 traded value", showlegend=False), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    fig.update_xaxes(tickformat="%H:%M")
    return _base(fig, 520)


def vol_fig(tr):
    """Average result and win rate by the volume class at the signal."""
    from volume_analysis import VOL_NAMES
    g = tr.groupby("vol_class").agg(trades=("net", "size"), win=("net", lambda x: 100 * (x > 0).mean()), avg=("net", "mean")).reset_index()
    g["class"] = g.vol_class.map(VOL_NAMES)
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Average net ₹ per trade", "Win rate %"))
    fig.add_trace(go.Bar(x=g["class"], y=g.avg, marker_color=[GREEN if v > 0 else RED for v in g.avg], text=g.trades.map(lambda n: f"{n} trades"),
                         textposition="outside", showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=g["class"], y=g.win, marker_color=BLUE, showlegend=False), row=1, col=2)
    return _base(fig, 320)


def rv_hist(tr):
    fig = go.Figure()
    for nm, sub, colr in (("winners", tr[tr.net > 0], GREEN), ("losers", tr[tr.net <= 0], RED)):
        fig.add_trace(go.Histogram(x=sub.rv.clip(upper=5), name=nm, marker_color=colr, opacity=0.6, nbinsx=30))
    fig.update_layout(barmode="overlay")
    return _base(fig, 300, "Relative volume at the signal (capped at 5x)")


def sensitivity_fig(grid, sl_method):
    g = grid[grid.sl_method == sl_method]
    p = g.pivot_table(index="min_score", columns="rr", values="net ₹")
    z = p.to_numpy(float)
    lim = np.nanmax(np.abs(z)) if np.isfinite(z).any() else 1
    fig = go.Figure(go.Heatmap(z=z, x=[f"1:{c:g}" for c in p.columns], y=[str(i) for i in p.index], zmid=0, zmin=-lim, zmax=lim,
                               colorscale=[[0, RED], [0.5, "#ffffff"], [1, GREEN]], texttemplate="%{z:,.0f}", showscale=False))
    fig.update_xaxes(title="risk : reward")
    fig.update_yaxes(title="minimum score", type="category")
    return _base(fig, 300, f"Net ₹ over the whole period, stop = {sl_method} (in-sample: for sensitivity only)")

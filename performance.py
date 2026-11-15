"""Performance numbers, equity and drawdown, group tables, out-of-sample split and walk-forward optimisation."""
import itertools

import numpy as np
import pandas as pd

from config import with_changes


def _streak(win):
    best_w = best_l = cur = 0
    last = None
    for w in win:
        cur = cur + 1 if w == last else 1
        last = w
        if w:
            best_w = max(best_w, cur)
        else:
            best_l = max(best_l, cur)
    return best_w, best_l


def summarize(tr):
    """All headline numbers in rupees (after costs) and points."""
    if tr is None or tr.empty:
        return {"Total trades": 0}
    w, l_ = tr[tr.net > 0], tr[tr.net <= 0]
    eq = tr.net.cumsum()
    dd = eq - eq.cummax().clip(lower=0)
    gp, gl = w.net.sum(), -l_.net.sum()
    sw, sl = _streak((tr.net > 0).tolist())
    days = tr.day.nunique()
    return {
        "Total trades": len(tr), "Winning trades": len(w), "Losing trades": len(l_),
        "Win rate %": round(100 * len(w) / len(tr), 1),
        "Gross profit ₹": round(gp, 0), "Gross loss ₹": round(-gl, 0), "Net profit ₹": round(tr.net.sum(), 0),
        "Costs paid ₹": round(tr.costs.sum(), 0), "Net points (before costs)": round(tr.points.sum(), 1),
        "Average profit per trade ₹": round(tr.net.mean(), 0),
        "Average win ₹": round(w.net.mean(), 0) if len(w) else 0, "Average loss ₹": round(l_.net.mean(), 0) if len(l_) else 0,
        "Profit factor": round(gp / gl, 2) if gl > 0 else float("inf"),
        "Max drawdown ₹": round(dd.min(), 0),
        "Risk-reward achieved": round(w.net.mean() / -l_.net.mean(), 2) if len(w) and len(l_) and l_.net.mean() < 0 else np.nan,
        "Average R multiple": round(tr.r_multiple.mean(), 2) if tr.r_multiple.notna().any() else np.nan,
        "Max consecutive wins": sw, "Max consecutive losses": sl, "Trading days with a trade": days,
        "t-stat (per day)": round(_tstat(tr.groupby("day").net.sum()), 2),
    }


def _tstat(x):
    x = np.asarray(x, float)
    return x.mean() / (x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 2 and x.std(ddof=1) > 0 else 0.0


def equity_curve(tr):
    if tr.empty:
        return pd.DataFrame(columns=["time", "equity", "drawdown"])
    e = pd.DataFrame({"time": tr.exit_time, "equity": tr.net.cumsum()})
    e["drawdown"] = e.equity - e.equity.cummax().clip(lower=0)
    return e


def group_table(tr, key):
    if tr.empty:
        return pd.DataFrame()
    g = tr.groupby(key)
    out = pd.DataFrame({"trades": g.size(), "win %": g.net.apply(lambda x: 100 * (x > 0).mean()).round(1),
                        "net ₹": g.net.sum().round(0), "avg ₹": g.net.mean().round(0), "points": g.points.sum().round(1),
                        "profit factor": g.net.apply(lambda x: round(x[x > 0].sum() / -x[x <= 0].sum(), 2) if (x <= 0).any() and x[x <= 0].sum() < 0 else np.inf)})
    return out.reset_index()


def monthly_table(tr):
    if tr.empty:
        return pd.DataFrame()
    d = pd.to_datetime(tr.day)
    p = tr.assign(Year=d.dt.year, Month=d.dt.month).pivot_table(index="Year", columns="Month", values="net", aggfunc="sum")
    p = p.reindex(columns=range(1, 13))
    p.columns = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    p["Total"] = p.sum(axis=1)
    return p.round(0)


def dow_table(tr):
    if tr.empty:
        return pd.DataFrame()
    t = tr.assign(weekday=pd.to_datetime(tr.day).dt.day_name())
    out = group_table(t, "weekday")
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
    return out.set_index("weekday").reindex([x for x in order if x in set(out.weekday)]).reset_index()


def split_in_out(tr, cutoff):
    """Numbers before and after a cut-off date: tune on the first part, judge on the second."""
    if tr.empty or cutoff is None:
        return None
    d = pd.to_datetime(tr.day)
    a, b = tr[d < pd.Timestamp(cutoff)], tr[d >= pd.Timestamp(cutoff)]
    keys = ["Total trades", "Win rate %", "Net profit ₹", "Average profit per trade ₹", "Profit factor", "Max drawdown ₹", "t-stat (per day)"]
    sa, sb = summarize(a), summarize(b)
    return pd.DataFrame({"In-sample (before)": [sa.get(k, 0) for k in keys], "Out-of-sample (after)": [sb.get(k, 0) for k in keys]}, index=keys)


GRID = {"rr": [1.0, 1.5, 2.0, 3.0], "sl_method": ["signal", "swing", "fixed"], "min_score": [50, 65, 80]}


def walk_forward(run, base, grid=GRID, min_trades=20, progress=None):
    """For every combination of the grid: one backtest over the whole period. Then for each test year,
    pick the combination that did best in all EARLIER years only and record what it did in the test year.
    `run(settings)` returns a trades DataFrame. Returns (per-year table, stitched out-of-sample trades, grid results)."""
    combos = list(itertools.product(*grid.values()))
    res = {}
    for n_, vals in enumerate(combos):
        if progress:
            progress(n_, len(combos))
        st = with_changes(base, **dict(zip(grid.keys(), vals)))
        tr = run(st)
        res[vals] = tr.assign(year=pd.to_datetime(tr.day).dt.year) if len(tr) else tr
    years = sorted({y for t in res.values() if len(t) for y in t.year.unique()})
    rows, oos = [], []
    for Y in years[1:]:
        best, best_net = None, -np.inf
        for vals, t in res.items():
            if not len(t):
                continue
            past = t[t.year < Y]
            if len(past) >= min_trades and past.net.sum() > best_net:
                best, best_net = vals, past.net.sum()
        if best is None:
            continue
        t = res[best]
        cur = t[t.year == Y]
        oos.append(cur)
        rows.append({"test year": Y, **dict(zip(grid.keys(), best)), "chosen on years before (net ₹)": round(best_net, 0),
                     "test year trades": len(cur), "test year net ₹": round(cur.net.sum(), 0),
                     "test year win %": round(100 * (cur.net > 0).mean(), 1) if len(cur) else 0})
    grid_tab = pd.DataFrame([{**dict(zip(grid.keys(), v)), "trades": len(t), "net ₹": round(t.net.sum(), 0) if len(t) else 0,
                              "profit factor": summarize(t).get("Profit factor", 0) if len(t) else 0} for v, t in res.items()])
    return pd.DataFrame(rows), (pd.concat(oos) if oos else pd.DataFrame()), grid_tab

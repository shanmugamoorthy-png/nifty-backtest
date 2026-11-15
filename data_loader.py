"""Data: NIFTY 1-min candles and NIFTY 50 traded value (the "volume") from Dhan, or your own CSV.
Stored in SQLite so the 5-year download happens only once. 5- and 15-min candles are built from the 1-min ones."""
import csv
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

from config import DB_PATH, DHAN_BASE, IST, NIFTY50, NIFTY_SECURITY_ID, SCRIP_MASTER, SESSION_END, SESSION_START

LOCK = threading.Lock()
EPOCH = pd.Timestamp("1970-01-01")


# ----------------------------------------------------------------------------------------------------------- storage
def db():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.execute("CREATE TABLE IF NOT EXISTS nifty1m(ts INTEGER PRIMARY KEY, o REAL, h REAL, l REAL, c REAL)")
    c.execute("CREATE TABLE IF NOT EXISTS volstock(sym TEXT, ts INTEGER, val REAL, PRIMARY KEY(sym, ts))")
    c.execute("CREATE TABLE IF NOT EXISTS vol5(ts INTEGER PRIMARY KEY, val REAL, n INTEGER)")
    c.execute("CREATE TABLE IF NOT EXISTS done(series TEXT, win TEXT, PRIMARY KEY(series, win))")
    return c


def to_min(dt):
    """IST clock time -> integer minutes (the IST clock is stored as if it were UTC, so no time-zone maths later)."""
    return int((pd.Timestamp(dt) - EPOCH) // pd.Timedelta(minutes=1))


def from_min(m):
    return EPOCH + pd.to_timedelta(np.asarray(m, dtype="int64"), unit="m")


# ----------------------------------------------------------------------------------------------------------- Dhan
def dhan_post(path, body):
    cid, tok = os.getenv("DHAN_CLIENT_ID"), os.getenv("DHAN_ACCESS_TOKEN")
    if not (cid and tok):
        raise RuntimeError("Set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN (Render: Environment tab)")
    hd = {"access-token": tok, "client-id": cid, "Content-Type": "application/json", "Accept": "application/json"}
    for _ in range(4):
        r = requests.post(DHAN_BASE + path, json=body, timeout=60, headers=hd)
        if r.status_code != 429:
            break
        time.sleep(3)
    if not r.ok:
        raise RuntimeError(f"HTTP {r.status_code} {r.text[:200]}")
    return r.json()


def fetch_intraday(sid, segment, instrument, interval, d0, d1):
    """One Dhan intraday request (at most 90 days). Returns rows (IST datetime, open, high, low, close, volume)."""
    body = {"securityId": str(sid), "exchangeSegment": segment, "instrument": instrument, "interval": str(interval),
            "oi": False, "fromDate": d0.strftime("%Y-%m-%d"), "toDate": (d1 + timedelta(days=1)).strftime("%Y-%m-%d")}
    j = dhan_post("/charts/intraday", body)
    j = j.get("data") or j
    ts = j.get("timestamp") or []
    if not ts:
        return []
    off = 315532800 if datetime.fromtimestamp(ts[0], IST).year < 2020 else 0      # some replies count from 1980
    vol = j.get("volume") or [0] * len(ts)
    out = []
    for i, t_ in enumerate(ts):
        x = datetime.fromtimestamp(t_ + off, IST).replace(tzinfo=None)
        if d0 <= x.date() <= d1 and SESSION_START <= x.hour * 60 + x.minute < SESSION_END:
            out.append((x, j["open"][i], j["high"][i], j["low"][i], j["close"][i], vol[i] or 0))
    return out


def equity_ids(symbols):
    """NSE equity security ids for the given symbols, read from Dhan's public instrument list."""
    r = requests.get(SCRIP_MASTER, stream=True, timeout=180)
    rd = csv.reader(x.decode("utf-8", "ignore") for x in r.iter_lines())
    hd = next(rd)
    ix = {n: hd.index(n) for n in ("SEM_EXM_EXCH_ID", "SEM_SEGMENT", "SEM_SMST_SECURITY_ID", "SEM_TRADING_SYMBOL", "SEM_SERIES")}
    top, want, out = max(ix.values()), set(symbols), {}
    for row in rd:
        if len(row) > top and row[ix["SEM_EXM_EXCH_ID"]] == "NSE" and row[ix["SEM_SEGMENT"]] == "E" \
                and row[ix["SEM_SERIES"]] == "EQ" and row[ix["SEM_TRADING_SYMBOL"]] in want:
            out[row[ix["SEM_TRADING_SYMBOL"]]] = row[ix["SEM_SMST_SECURITY_ID"]]
    return out


def windows(years, days=85):
    end = datetime.now(IST).date()
    a = end - timedelta(days=int(365.25 * years))
    while a <= end:
        b = min(a + timedelta(days=days - 1), end)
        yield a, b
        a = b + timedelta(days=1)


def download_job(state, years=5.0):
    """Background job: NIFTY 1-min candles, then each NIFTY 50 stock's 5-min traded value. Safe to re-run:
    finished windows are skipped, the latest window is refreshed."""
    state.update(running=True, msg="starting...", done=0, total=0, errors=[])
    try:
        today = datetime.now(IST).date()
        with LOCK:
            c = db()
            have = {(a, b) for a, b in c.execute("SELECT series, win FROM done")}
            c.close()
        state["msg"] = "reading the stock list..."
        ids = equity_ids(NIFTY50)
        missing = [s for s in NIFTY50 if s not in ids]
        if missing:
            state["errors"].append(f"not found in Dhan's list (skipped): {', '.join(missing)}")
        jobs = [("NIFTY1", NIFTY_SECURITY_ID, "IDX_I", "INDEX", 1)] + [(s, ids[s], "NSE_EQ", "EQUITY", 5) for s in NIFTY50 if s in ids]
        wins = list(windows(years))
        state["total"] = len(jobs) * len(wins)
        bad = 0
        for name, sid, seg, ins, iv in jobs:
            for a, b in wins:
                state["done"] += 1
                key = a.strftime("%Y-%m-%d")
                if (name, key) in have and b < today - timedelta(days=3):
                    continue
                state["msg"] = f"{name} {a:%d %b %Y} to {b:%d %b %Y}"
                try:
                    rows = fetch_intraday(sid, seg, ins, iv, a, b)
                    bad = 0
                except Exception as e:
                    bad += 1
                    if len(state["errors"]) < 8:
                        state["errors"].append(f"{name} {key}: {str(e)[:150]}")
                    if "401" in str(e) or "DH-906" in str(e) or "DHAN_" in str(e):
                        state["msg"] = "stopped: Dhan token problem. Update DHAN_ACCESS_TOKEN and press the button again."
                        return
                    if bad >= 40:
                        state["msg"] = "stopped: too many errors in a row"
                        return
                    continue
                with LOCK:
                    c = db()
                    if name == "NIFTY1":
                        c.executemany("INSERT OR REPLACE INTO nifty1m VALUES(?,?,?,?,?)",
                                      [(to_min(x), o, h, l, cl) for x, o, h, l, cl, v in rows])
                    else:
                        c.executemany("INSERT OR REPLACE INTO volstock VALUES(?,?,?)",
                                      [(name, to_min(x), float(v) * float(cl) / 1e7) for x, o, h, l, cl, v in rows])
                    c.execute("INSERT OR IGNORE INTO done VALUES(?,?)", (name, key))
                    c.commit()
                    c.close()
                time.sleep(0.25)
        state["msg"] = "adding up the 50 stocks' traded value..."
        with LOCK:
            c = db()
            c.execute("DELETE FROM vol5")
            c.execute("INSERT INTO vol5 SELECT ts, SUM(val), COUNT(*) FROM volstock GROUP BY ts")
            c.commit()
            c.close()
        state["msg"] = "finished"
    except Exception as e:
        state["msg"] = f"failed: {str(e)[:200]}"
    finally:
        state["running"] = False


def stored_summary():
    with LOCK:
        c = db()
        a = c.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM nifty1m").fetchone()
        b = c.execute("SELECT COUNT(*), MIN(ts), MAX(ts), AVG(n) FROM vol5").fetchone()
        c.close()
    f = lambda m: "-" if m is None else f"{from_min([m])[0]:%d %b %Y}"
    return (f"NIFTY 1-min candles: {a[0]:,} ({f(a[1])} to {f(a[2])}).  "
            f"NIFTY 50 volume (5-min): {b[0]:,} bars ({f(b[1])} to {f(b[2])}, about {b[3] or 0:.0f} stocks per bar).")


# ----------------------------------------------------------------------------------------------------------- loading
def load_minutes():
    with LOCK:
        c = db()
        d = pd.read_sql("SELECT ts, o, h, l, c FROM nifty1m ORDER BY ts", c)
        c.close()
    d["dt"] = from_min(d.pop("ts"))
    d["v"] = np.nan
    return d[["dt", "o", "h", "l", "c", "v"]]


def load_volume5():
    with LOCK:
        c = db()
        d = pd.read_sql("SELECT ts, val FROM vol5 ORDER BY ts", c)
        c.close()
    d["dt"] = from_min(d.pop("ts"))
    return d.rename(columns={"val": "v"})[["dt", "v"]]


def parse_csv(file):
    """Your own data: columns datetime, open, high, low, close, volume (any interval). Times are taken as IST
    unless they carry a time zone, which is then converted to IST."""
    d = pd.read_csv(file)
    d.columns = [x.strip().lower() for x in d.columns]
    need = ["datetime", "open", "high", "low", "close"]
    miss = [x for x in need if x not in d.columns]
    if miss:
        raise ValueError(f"CSV is missing column(s): {', '.join(miss)}")
    dt = pd.to_datetime(d["datetime"], errors="coerce", utc=False)
    if getattr(dt.dt, "tz", None) is not None:
        dt = dt.dt.tz_convert(IST).dt.tz_localize(None)
    out = pd.DataFrame({"dt": dt, "o": pd.to_numeric(d["open"], errors="coerce"), "h": pd.to_numeric(d["high"], errors="coerce"),
                        "l": pd.to_numeric(d["low"], errors="coerce"), "c": pd.to_numeric(d["close"], errors="coerce"),
                        "v": pd.to_numeric(d["volume"], errors="coerce") if "volume" in d.columns else np.nan})
    return out.dropna(subset=["dt", "o", "h", "l", "c"])


def clean(df):
    """Sort, remove duplicate candles, keep market hours on weekdays. Returns (data, quality report dict)."""
    n0 = len(df)
    df = df.sort_values("dt").drop_duplicates("dt", keep="last")
    dup = n0 - len(df)
    m = df.dt.dt.hour * 60 + df.dt.dt.minute
    df = df[(m >= SESSION_START) & (m < SESSION_END) & (df.dt.dt.weekday < 5)].reset_index(drop=True)
    step = int(df.dt.diff().dt.total_seconds().div(60).dropna().clip(lower=1).median()) if len(df) > 2 else 1
    per_day = (SESSION_END - SESSION_START) // max(step, 1)
    cnt = df.groupby(df.dt.dt.date).size()
    short = cnt[cnt < per_day * 0.95]
    bad = df[(df.h < df[["o", "c"]].max(axis=1)) | (df.l > df[["o", "c"]].min(axis=1))]
    rep = {"candles": len(df), "interval_min": step, "days": int(cnt.size), "duplicates_removed": int(dup),
           "outside_hours_removed": int(n0 - dup - len(df)), "expected_per_day": per_day,
           "days_with_missing_candles": int(short.size), "missing_candles": int((per_day - cnt).clip(lower=0).sum()),
           "bad_high_low_rows": int(len(bad)), "short_days": [str(x) for x in short.index[:15]]}
    return df, rep


def resample(df, tf):
    """Candles of tf minutes from smaller candles, aligned to 09:15 each day. Volume is added up."""
    m = df.dt.dt.hour * 60 + df.dt.dt.minute
    day = df.dt.dt.normalize()
    k = day + pd.to_timedelta(((m - SESSION_START) // tf) * tf + SESSION_START, unit="m")
    g = df.groupby(k, sort=True)
    out = pd.DataFrame({"o": g.o.first(), "h": g.h.max(), "l": g.l.min(), "c": g.c.last(),
                        "v": g.v.sum(min_count=1)})
    out.index.name = "dt"
    return out.reset_index()


def build_bars(minutes, vol5, tf):
    """Trading candles of tf minutes (5 or 15) with NIFTY 50 traded value as volume."""
    bars = resample(minutes, tf)
    if vol5 is not None and len(vol5):
        vb = resample(vol5.assign(o=0.0, h=0.0, l=0.0, c=0.0), tf)[["dt", "v"]]
        bars = bars.drop(columns="v").merge(vb, on="dt", how="left")
    bars["day"] = bars.dt.dt.normalize()
    bars["mins"] = bars.dt.dt.hour * 60 + bars.dt.dt.minute
    return bars.reset_index(drop=True)

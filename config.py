"""Settings and constants for the NIFTY price-action backtester. Every number here can be changed from the dashboard."""
import os
from dataclasses import dataclass, asdict, replace
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
SESSION_START = 9 * 60 + 15        # 09:15, minutes after midnight
SESSION_END = 15 * 60 + 30         # 15:30
NIFTY_SECURITY_ID = "13"           # Dhan id of the NIFTY 50 index (segment IDX_I)
DB_PATH = os.environ.get("DB_PATH", "nifty_bt.db")
DHAN_BASE = "https://api.dhan.co/v2"
SCRIP_MASTER = "https://images.dhan.co/api-data/api-scrip-master.csv"

# The 50 index stocks whose 5-min traded value (volume x price) is added up as NIFTY "volume".
# The index itself has no traded volume. Edit this list if the index members change.
NIFTY50 = ["ADANIENT", "ADANIPORTS", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV",
           "BEL", "BHARTIARTL", "CIPLA", "COALINDIA", "DRREDDY", "EICHERMOT", "ETERNAL", "GRASIM", "HCLTECH", "HDFCBANK",
           "HDFCLIFE", "HINDALCO", "HINDUNILVR", "ICICIBANK", "INDIGO", "INFY", "ITC", "JIOFIN", "JSWSTEEL", "KOTAKBANK",
           "LT", "M&M", "MARUTI", "MAXHEALTH", "NESTLEIND", "NTPC", "ONGC", "POWERGRID", "RELIANCE", "SBILIFE", "SBIN",
           "SHRIRAMFIN", "SUNPHARMA", "TATACONSUM", "TMPV", "TATASTEEL", "TCS", "TECHM", "TITAN", "TRENT", "ULTRACEMCO", "WIPRO"]

ALL_STRATEGIES = {
    "ORB": "A. Opening range breakout",
    "PDHL": "B. Previous day high/low breakout",
    "SR": "C. Support/resistance reversal",
    "SD": "D. Supply/demand zone retest",
    "CANDLE": "E. Candlestick price action",
    "MS": "F. Market structure (BOS / CHoCH)",
    "MODEL": "G. NIFTY price model (walk-forward)",
}


@dataclass(frozen=True)
class Settings:
    # what to trade
    timeframe: int = 5                  # trading candles, minutes (5 or 15)
    htf: int = 15                       # higher timeframe for confirmation, minutes
    strategies: tuple = ("ORB", "PDHL", "SR", "SD", "CANDLE", "MS")
    entry_mode: str = "close"           # "close" = at the signal candle close, "next_open" = next candle open
    min_score: int = 65                 # 80+ very strong, 65+ strong, 50+ moderate
    strict: bool = False                # also require: structure agrees, bullish/bearish candle, volume >= moderate
    max_trades_day: int = 3
    first_entry: int = 9 * 60 + 30      # no entries before 09:30
    last_entry: int = 14 * 60 + 45      # no new entries after 14:45
    square_off: int = 15 * 60 + 15      # everything closed at 15:15
    # stop loss and target
    sl_method: str = "signal"           # "signal" candle, last "swing", "fixed" points, or "none"
    sl_fixed_pts: float = 30.0
    sl_buffer_pts: float = 2.0
    sl_min_pts: float = 10.0
    sl_max_pts: float = 80.0
    rr: float = 2.0                     # target = rr x risk; 0 = no target
    breakeven: bool = True
    be_at_r: float = 1.0
    trailing: bool = False
    trail_at_r: float = 1.0
    partial: bool = False
    partial_at_r: float = 1.0
    partial_frac: float = 0.5
    max_hold_bars: int = 0              # time exit after this many trading candles; 0 = off
    # volume
    vol_mode: str = "candles"           # "candles" = average of the previous 20 candles, "time" = same time of day, previous 20 days
    vol_lookback: int = 20
    vol_strong: float = 1.5
    vol_moderate: float = 1.2
    # structure and levels
    pivot_k: int = 3                    # a swing high/low needs k candles on each side (known only k candles later)
    sr_lookback_days: int = 5
    sr_tol_pct: float = 0.08            # price is "at" a level within this % of it
    impulse_atr: float = 1.8            # impulsive candle = range >= this x ATR
    base_max_bars: int = 3
    # money
    lot_size: int = 65
    lots: int = 1
    brokerage_flat: float = 20.0        # per executed order
    brokerage_pct: float = 0.03         # or this % of order value, whichever is lower
    stt_sell_pct: float = 0.05          # futures STT, sell side
    exch_pct: float = 0.00183           # NSE futures transaction charge
    sebi_per_cr: float = 10.0
    stamp_buy_pct: float = 0.002
    gst_pct: float = 18.0
    slippage_pts: float = 1.0           # added against you on every fill

    def to_dict(self):
        return asdict(self)


DEFAULTS = Settings()


def with_changes(s: Settings, **kw) -> Settings:
    return replace(s, **kw)

# NIFTY price-action backtester

Research tool only: it backtests, it never trades and never sends alerts. Your live NIFTY dashboard and its paper log are separate and untouched.

## What it does
- **Data:** NIFTY 50 index 1-minute candles from Dhan (up to 5 years). 5- and 15-minute candles are built from them.
- **Volume:** the index has no traded volume, so NIFTY "volume" = 5-minute traded value (volume × price) of the 50 index stocks added together. Edit the list in `config.py` if the index changes.
- **Strategies (each can be switched on/off):** A opening-range breakout, B previous-day high/low breakout, C support/resistance rejection, D supply/demand retest, E candlestick patterns, F market structure (BOS/CHoCH), G the NIFTY price model from your dashboard (trained year by year on earlier years only).
- **Score 0-100:** structure 20, setup 25, volume 20, support/resistance 15, higher timeframe 20. 80+ very strong, 65+ strong, 50+ moderate, below 50 avoid.
- **Exits:** stop at signal candle / last swing / fixed points / none; target 1:1, 1:1.5, 1:2, 1:3 or none; break-even, trailing, 50% partial booking, time exit, square-off 15:15. Exits are checked on 1-minute candles; if a stop and a target fall in the same minute, the stop is assumed first.
- **Costs:** NIFTY futures charges (brokerage, STT 0.05% on sell, exchange, SEBI, stamp duty, GST) plus slippage, all editable. At NIFTY ≈ 25,000 one lot (65) costs about ₹966 per round trip ≈ 15 points (mostly STT), before slippage.
- **No look-ahead:** swing points count only after the candles that confirm them have closed; the 15-min trend uses only closed 15-min candles. Tested by changing all future data and checking that no earlier signal changed.
- **Validation:** in-sample vs out-of-sample split by date; walk-forward optimisation (36 combinations, each year judged with the combination chosen on earlier years only); a sensitivity map.

## Deploy on Render (like your other two services)
1. Create a new GitHub repository, e.g. `nifty-backtest`, and add every file of this folder (13 files) at the top level.
2. Render → New → Web Service → pick the repository.
   - Build command: `pip install -r requirements.txt`
   - Start command: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
   - Instance type: Starter.
3. Environment variables: `PYTHON_VERSION` = `3.12.8`, `DHAN_CLIENT_ID`, `DHAN_ACCESS_TOKEN` (same as your other services), `DB_PATH` = `/data/nifty_bt.db`.
4. Disks → Add disk: mount path `/data`, size 1 GB (the 5-year data uses about 250-350 MB).
5. Open the page → sidebar → **Download / update data**. 5 years takes about 20-40 minutes; press **Refresh progress** to see where it is. The download continues if you close the page; a deploy or restart stops it, and pressing the button again continues where it stopped.

## Run on a Windows PC instead
1. Install Python 3.12 from python.org (tick "Add python.exe to PATH").
2. Open Command Prompt in this folder and run `pip install -r requirements.txt`.
3. `set DHAN_CLIENT_ID=your_id` and `set DHAN_ACCESS_TOKEN=your_token` (or skip and upload a CSV).
4. `streamlit run app.py` — it opens in your browser. Data is saved in `nifty_bt.db` in the same folder.

## Your own CSV
Columns `datetime, open, high, low, close, volume`. Any interval up to the trading timeframe (1-min is best: exits are then checked minute by minute). Times are taken as IST unless they carry a time zone. Duplicates are removed, candles outside 09:15-15:30 dropped, missing candles counted in the Data tab.

## How to read the results honestly
- Trust the **out-of-sample** column and the **walk-forward** result, not the best-looking setting.
- Every change you make while looking at results is a hidden test; the more you try, the more likely a good number is luck.
- Earlier tests in this project found no NIFTY breakout or pattern edge large enough to beat costs; expect most combinations here to lose after costs.
- Index backtests assume you trade NIFTY futures at the index price; the futures basis and real fills differ a little.
- Not investment advice.

## Files
`app.py` dashboard · `config.py` settings · `data_loader.py` Dhan/CSV/SQLite, candle building, quality checks · `indicators.py` ATR and helpers · `price_action.py` candlestick patterns · `market_structure.py` swings, BOS/CHoCH, levels, supply/demand · `volume_analysis.py` relative volume, rejection, divergence · `strategies.py` signals A-G and the score · `risk_management.py` stops, targets, costs · `backtest_engine.py` the engine · `performance.py` statistics and walk-forward · `charts.py` Plotly charts · `requirements.txt`.

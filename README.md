# Cronus Trading Bot — "Titan Trend" strategy

A production-grade, ML-augmented Telegram trading signal bot for futures,
running the **Titan Trend** strategy (trend + momentum + volume-proxy
confluence, ATR-based risk management, ML-scaled confidence). Built and
validated on CME E-mini/Micro E-mini ES, NQ, MES, MNQ data, but the data
loader and feature pipeline work on any OHLC(V) CSV — see
`src/data_layer/loader.py`. The strategy's display name is set in
`config/config.yaml` under `project.strategy_name` and appears on every
Telegram signal message.

**Read `reports/eda_summary.md` first.** It documents the actual data
quality issues found in the provided history (no real volume, thin sample
size on two symbols) and the design decisions made in response — this
README assumes you've seen that.

## What's in here

```
config/config.yaml       All strategy/ML/backtest/bot parameters
.env.example              Copy to .env and fill in secrets (never commit .env)
data/raw/                 Your historical CSVs
src/
  data_layer/              CSV loading, format auto-detection, data-quality checks
  features/                Technical indicators, regime detection, feature engineering
  strategy/                Rule-based entry/exit logic, risk management/position sizing
  ml/                      Leak-free labeling, walk-forward validation, model training
  backtest/                Backtest engine, performance metrics, charts/reports
  telegram_bot/            Telegram client, message formatting, bot orchestrator
  utils/                   Config loading, logging
scripts/
  train.py                 Train + evaluate all ML models, save the best one per symbol
  backtest.py               Run the full backtest, generate reports/*.png and *.md
  live_signal.py             Check for a new signal and broadcast it (--once or --forever)
tests/                      pytest unit tests (indicators, loader, strategy, ML, metrics)
reports/                     Generated backtest reports, charts, and the EDA summary
models/                      Trained model files (.joblib) + training summaries (.json)
Dockerfile, docker-compose.yml, requirements.txt
```

## 1. Setup

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env: set TELEGRAM_BOT_TOKEN (from @BotFather) and TELEGRAM_CHAT_IDS
```

Run the tests to confirm everything imports and works in your environment:

```bash
pytest
```

## 2. Train the ML models

```bash
python scripts/train.py
```

This builds features for every symbol, labels them with a leak-free
triple-barrier method, walk-forward validates Random Forest / Gradient
Boosting / Hist Gradient Boosting / XGBoost / LightGBM / CatBoost (whichever
of the last three are installed — they're optional dependencies, see
`src/ml/models.py`), picks the best by out-of-sample ROC-AUC, refits it on
full history, and saves it to `models/`. Per-symbol summaries land in
`models/<symbol>_training_summary.json`.

Only symbols marked `ml_enabled: true` in `config/config.yaml` are trained
(by default: ES and NQ — see the EDA summary for why MES/MNQ are excluded).

## 3. Backtest

```bash
python scripts/backtest.py                 # all symbols, ML-augmented where available
python scripts/backtest.py --symbol ES      # one symbol
python scripts/backtest.py --no-ml          # rules-only, no ML blending
```

Produces `reports/<symbol>_backtest_report.md`, equity curve / drawdown /
trade P&L charts, and a full trade log CSV. The ML blend in the backtest
uses genuinely out-of-sample walk-forward probabilities (see
`src/ml/walkforward.py::generate_oos_probabilities`) — it never scores a
bar with a model that was trained on data including or after that bar.

## 4. Run the live bot

```bash
python scripts/live_signal.py --once        # single check, good for cron/CI
python scripts/live_signal.py --forever      # long-running loop
```

The bundled data source (`CsvReplayDataSource`) evaluates your historical
CSVs' most recent bar, so you can validate the whole pipeline — including
the actual Telegram message — before a live feed exists. To go live, implement
a real feed in `src/data_layer/live_feed.py` and pass it to `SignalBot`.

### Deploying on a schedule

This mirrors the polling-schedule pattern already used elsewhere in this
project (GitHub Actions calling the bot on an interval matching the
timeframe): add a workflow that runs `python scripts/live_signal.py --once`
every 5 or 15 minutes (matching your fastest symbol's timeframe), or use the
long-running `--forever` mode in Docker/a VPS instead.

## 5. Docker

```bash
docker compose build
docker compose up -d cronus-bot                 # long-running signal bot
docker compose run --rm trainer                 # one-off training job
docker compose run --rm backtester               # one-off backtest job
```

## Important limitations to know before trading real money

1. **No real trade volume in the provided data.** VWAP and "volume spike"
   confirmation are computed from documented proxies (rolling typical price,
   and a range/body-based "participation proxy"), not genuine traded volume.
   Every place this matters is labeled in code and in signal messages
   ("volume-proxy", not "volume"). If you get a volume-inclusive data feed,
   the loader will automatically switch to true VWAP/volume-zscore — no
   code changes needed.
2. **The ML edge is real but modest.** Out-of-sample walk-forward ROC-AUC
   for the included models is in the ~0.51–0.53 range on ES — better than
   a coin flip, but far from a lock. This is normal for short-horizon
   futures direction prediction and is exactly why the ML model is used to
   *veto/scale confidence* on rule-based signals rather than trade alone.
   Trust the backtest numbers in `reports/`, not this paragraph, for the
   final word — and re-run `scripts/backtest.py` whenever you add data.
3. **MES and MNQ are rules-only.** Their history (about 2 and 7 weeks
   respectively) is too short to walk-forward validate an ML model
   credibly; extend their data and set `ml_enabled: true` in
   `config/config.yaml` once you have enough (a rough rule of thumb: at
   least the walk-forward config's `min_train_bars` plus several
   `test_window_bars` worth of *additional* history so more than one fold
   exists).
4. **Backtest costs are estimates.** `commission_per_trade` and
   `slippage_ticks` in `config.yaml` are reasonable placeholders, not your
   broker's real numbers — update them before trusting absolute P&L figures.
5. **This is a decision-support tool, not an execution system.** It
   recommends entries, stops, targets, and position sizing; it does not
   place orders. Nothing here is financial advice.

## Extending

- **Add a symbol/timeframe:** add raw file + entry to `config/config.yaml`'s
  `symbols:` list. The loader auto-detects the CSV's timestamp format and
  volume availability.
- **Add a data source:** any CSV with an epoch or ISO timestamp column and
  open/high/low/close (volume optional) works out of the box — see
  `src/data_layer/loader.py`'s column-alias table if your headers are named
  differently.
- **Tune the strategy:** every threshold lives in `config/config.yaml`
  under `strategy:` — no code changes needed for most adjustments.
- **Add a model:** implement it in `src/ml/models.py::get_available_models`
  following the existing optional-import pattern.

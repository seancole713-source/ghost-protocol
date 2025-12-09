Ghost Intelligence Workers

This folder contains four lightweight background workers started by the FastAPI app on startup:

- macro_brain_worker.py — Computes a [-100, 100] macro pressure index from index momentum, rates/FX, vol proxy, and news sentiment. Persists to SQLite and mirrors latest to Redis.
- liquidity_monitor.py — Captures DXY, TLT (yields proxy), VIX, and placeholder flow metrics. Persists to SQLite and mirrors latest to Redis.
- pattern_memory.py — Maintains a small analog memory in DuckDB with vectors derived from features (ret_1d, dist_avg, news, qty). Provides cosine-similarity search utilities.
- reflex_trainer.py — Periodically re-weights module influence based on forecast accuracy stored in forecast_48h. Persists module_weights in SQLite.

API surface:

- /api/ai/signals — Returns a compact snapshot of macro_pressure, liquidity, and module_weights for consumption by /ai/decide and forecasting.

Configuration (env):

- MACRO_BRAIN_REFRESH_S, LIQUIDITY_REFRESH_S, PATTERN_REFRESH_S, REFLEX_REFRESH_S
- WOLF_SQLITE_PATH (default data/wolf.db), GHOST_DUCKDB (default ghost.duckdb)
- REDIS_URL and REDIS_PREFIX (optional; JSON mirrors for latest signals)

Dependencies:

- duckdb, numpy, yfinance (optional but recommended)

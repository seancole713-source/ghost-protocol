# Changelog

## v0.3.0 - 2025-09-24 (Resilient Unified Cockpit Baseline)

### Added

- Resilient unified cockpit snapshot builder with LAST_GOOD_SNAPSHOT fallback.
- `flags.degraded` and top-level `degraded` (alias) plus `fail_reasons` array for


  graceful partial failures.

- Reintroduced `/api/cockpit/stream` SSE endpoint (continuous unified snapshot events).
- Runtime fault injection endpoint: `POST /api/cockpit/force_fail` for resilience tests.
- News Feed panel restored in cockpit UI; snapshot now includes `news` (relevant) and


  `news_all` collections.

- Example snapshot contract fixture: `docs/snapshot_contract_example_v0_3_0.json`.


### Changed

- Portfolio `cash` now an object `{stock, crypto, total}`; maintained


  backward-compatible numeric semantics via tests.

- Added alias `price` mirroring `px` for legacy consumers; retained `positions` alias of


  `rows`.

- Updated `docs/UI_BASELINE_CONTRACT.md` to lock schema & semantics for v0.3.0.
- Version bumped to 0.3.0 (baseline lock) in `pyproject.toml`.


### Deprecated

- Reliance on implicit numeric cash field (use `cash.total`).


### Fixed

- Eliminated panel desynchronization / intermittent 500s by centralizing snapshot build.
- Resolved missing News Feed regression in cockpit layout distribution (`templates/` +


  `ui_dist/`).

### Testing

- Added resilience test ensuring degraded snapshot surfaces prior good data with


  `flags.degraded=true` when forced failure occurs.

### Ops Notes

- Fallback path ensures UI never shows raw error state; always yields at least last good


  snapshot.

- Force-fail endpoint resets after single invocation, enabling deterministic test


  toggling without env restarts.

______________________________________________________________________

## v0.2.0 - 2025-09-24 (Observability v1 + Math/Invariants Overhaul)

### Added

- Prometheus metrics endpoint (`/metrics`) exposing real histogram/counter/gauge


  registry.

- Snapshot latency histogram `ghost_cockpit_snapshot_build_seconds` powering UI latency


  badge (toggle via `UI_LATENCY_BADGE=0`).

- Provider latency histogram `ghost_provider_fetch_seconds{provider}` (alpha, polygon,


  yf, sim).

- Snapshot failure counter `ghost_cockpit_snapshot_failures_total` with test flag


  `SNAP_FORCE_FAIL=1`.

- Dynamic gauges: `ghost_up`, `ghost_active`, `ghost_error_count`,


  `ghost_timestamp_seconds`.

- Diagnostics surfaces recent NAV invariant failures (up to 5) via


  `diagnostics.summary().invariants`.

- Unified single-oracle cockpit snapshot with correct PnL math & stale semantics.
- New tests: metrics, math invariants (NAV, clamp, parity, stale zeroing, invariant


  logging).

- UI stale badge + removal of client-side PnL calculations (server authoritative


  snapshot only).

### Changed

- Deprecated legacy fields retained (`price`,`current_price`) for backward


  compatibility; prefer `current`.

- PnL % clamp implemented (never < -100%).
- Stale rows: `current` set to null; mark value frozen at entry.
- NAV invariant logged as `invariant_fail` event when drift detected.


### Ops Notes

- Metric registration guarded (idempotent) to avoid duplicate series on reloads.
- SSE latency badge computes p50/p95 from last 30 client parse timings of cockpit SSE


  events.

- Production scrape: 15s interval recommended; add alerting on failure rate & p95


  latency.

- SIM first tick now forced fresh (not stale) for deterministic acceptance tests.


### Test Suite

- Post-merge: 20 passed, 2 skipped (math + metrics suites included).


______________________________________________________________________

______________________________________________________________________

## v0.1.0 - 2025-09-24 (Initial Cockpit Release)

### Highlights

- Unified `/api/cockpit` snapshot (single source of truth) with stable `snapshot_id`


  consumed by UI + SSE stream `/api/cockpit/stream`.

- Portfolio math correctness: `NAV = cash + Σ(qty * price)`; invariant enforced by


  tests.

- Position dedup + weighted average merge logic fixed (stable entry price calculation


  only from cost basis, deterministic across adds).

- Per-row PnL absolute + percentage (formatted to 6 decimals) within tolerance ≤ 1e-4;


  acceptance tests green.

- Deterministic SIM mode price generator for offline / rate-limited development.
- Redis persistence path: positions + cash survive restarts when `ENABLE_REDIS=1`.
- Diagnostics + alerts self-test endpoint and badge integration (log feed available for


  observability).

- Codespaces / container autostart on port 5000 (Uvicorn single worker) for instant dev


  spin-up.

- Weighted merge unit test + cockpit acceptance test both passing.


### Stability / Tooling

- Test suite: 14 passed, 2 skipped, 0 failed at release cut.
- Removed temporary debugging instrumentation (ring buffers, snapshot taps, debug


  endpoints).

- Introduced consistent 6-decimal PnL % formatting.


### Operational Notes

- External providers (Coingecko, Polygon, AlphaVantage, yfinance) are opportunistic; SIM


  mode recommended when offline.

- SSE stream keeps cockpit live; clients may reconnect using `Last-Event-ID` if desired


  (future enhancement).

### Next (Potential Roadmap)

- Price fetch batching + circuit breakers.
- Prometheus metrics and structured logging (OpenTelemetry).
- Webhook-based alert fanout (Slack/Discord) alongside Telegram.
- Order execution simulation layer (pending design).


______________________________________________________________________

Cut from branch `rescue/rollback` against `main`.

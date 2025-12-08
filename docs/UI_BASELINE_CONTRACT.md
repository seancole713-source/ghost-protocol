# UI BASELINE CONTRACT — FROZEN LAYOUT

**Status:** FROZEN (v0.3.0 baseline). The cockpit UI (HTML/CSS/JS structure) is locked.
Only backend/data logic may change unless an approved RFC explicitly unfreezes a file.

## Frozen Paths (No Structural Changes)

- `templates/**`
- `ui_dist/**`
- `static/**`
- Root HTML files: `index.html`, `markets.html`


Allowed edits ONLY:

- Bugfixes that do not add/remove DOM nodes or change class names / layout semantics.
- Hooking to *new optional* backend fields (must degrade gracefully if missing).
- Accessibility fixes (aria labels) with no visual shift.


NOT allowed without RFC:

- New panels, buttons, or layout grids
- Renaming classes used in CSS
- Removing existing IDs used by JS
- Changing table column order


## Single Source of Truth

All front-end panels MUST derive from **one** snapshot:

`GET /api/cockpit -> { snapshot_id, as_of, mode, status, flags, prices, portfolio, movers, heatmap, heatmap_obj,
outlook, signals, news, news_all, alerts_recent, events_recent, fail_reasons, kpis }`

Resilience policy:

*Always 200 with last good snapshot if current build fails (adds `flags.degraded=true` & appends `top_level:<Err>` in `fail_reasons`).* Only 503 if no successful snapshot has ever been built.

* `flags.any_stale` true if any asset price > TTL (sim vs live policy).


SSE: `/api/cockpit/stream` emits the *same* shape (identical keys).
Test hook: `POST /api/cockpit/force_fail` forces next build failure (resilience test).

## Portfolio Invariants

- `NAV = cash + Σ(qty * current)` (rows with `current == null` excluded from PnL contribution)
- Row: `pnl_abs = (current - entry) * qty`
- Row: `pnl_pct = (current - entry) / entry` (clamped >= -100%)
- Positions deduped by `(symbol, market/type)` with **weighted average entry**: `entry = Σ(price_i * qty_i) / Σ(qty_i)`
- No duplicate rows in UI table
- Stale: `current = null`, `stale = true`, mark_value frozen at entry


## Heatmap / Movers Consistency

- Prices & GPS values must come from the exact same oracle used in portfolio.
- No 0.00 placeholders — if unavailable: `price = null`, mark stale/offline.


## Snapshot Requirements

Baseline mandatory keys (core contract):

```text
[
  "snapshot_id", "as_of", "prices", "portfolio", "movers",
  "heatmap", "signals", "status"
]

```text

Extended baseline (v0.3.0) expectations:

```text

flags.degraded (bool)
fail_reasons (array)
prices[SYM].{price,px,stale,src,type,ts}
portfolio.cash.{stock,crypto,total}
portfolio.rows[*].{symbol,qty,entry,current?,stale,gps}
portfolio.positions (legacy alias list)
news (array) & news_all (array)
heatmap_obj.tiles mirrors heatmap
outlook.signals (root signals is alias)

```text

Compatibility: `price` retained while `px` becomes canonical. New consumers should read `px`.

## Diagnostics

- Expose only concise structured items (invariant failures, error_count, recent alerts)
- No raw full snapshot dumps


## Alerts

- Debounce self-test / housekeeping alerts (>=10 min separation)
- All alert sends append a lightweight event into diagnostics buffer


## Testing Gate (Must Pass Before Merge)

1. Acceptance cockpit math (weighted merge, NAV invariant)
2. Parity (movers/heatmap prices match snapshot oracle)
3. SSE consistency (two consecutive ticks consistent math, new snapshot_id)
4. Snapshot contract test (schema keys present)
5. Resilience test (forced failure returns degraded snapshot + new fail_reasons entry)


## CI Guard

A workflow blocks PRs that modify frozen UI files. To change UI:

1. Open RFC issue with rationale + mock
2. Get approval
3. Add an allowlist override in PR description


## RFC Template (open an Issue)

```text

Title: UI Change Proposal – <short>
Problem: <what's wrong>
Proposed Change: <succinct summary>
Impact: <why necessary>
Alternatives: <other options considered>
Rollback: <how to revert>

```text

---
Maintaining a frozen UI isolates regressions to backend math & data flows, preserving trust in displayed NAV/PnL while
adding graceful degradation (no panel disappears, placeholders instead of errors).

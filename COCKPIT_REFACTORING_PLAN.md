# Cockpit.html JavaScript Refactoring Plan

## Current State

`cockpit.html` contains **~1900 lines**with**~900 lines of inline JavaScript**split
across multiple `<script>` blocks. This creates maintenance challenges:

- Difficult to debug with browser DevTools (no source maps)
- Hard to test individual functions
- Linter warnings due to template literal complexity
- No code reuse across pages
- Version control diffs are noisy (HTML + JS mixed)

## Proposed Modular Structure

### File Organization

```text
static/
├── js/
│   ├── ghost-cockpit-core.js       # Core utilities (api, fmt, el helpers)
│   ├── ghost-cockpit-ui.js         # UI state management (buttons, badges, loaders)
│   ├── ghost-cockpit-data.js       # Data fetching (portfolio, heatmap, news)
│   ├── ghost-cockpit-charts.js     # Chart rendering (forecast overlay, two-line)
│   ├── ghost-cockpit-stages.js     # Stage 1-5 loaders (world context, accuracy, regime, etc.)
│   ├── ghost-cockpit-alerts.js     # Corporate action banner, provider diagnostics
│   └── ghost-cockpit-init.js       # Main initialization and event wiring
templates/
└── cockpit.html                     # Minimal HTML structure + script includes

```text

### Module Breakdown

#### 1.**ghost-cockpit-core.js** (~50 lines)

Core utilities and constants:

```javascript

// Constants
export const LATENCY_ENABLED = (window.UI_LATENCY_BADGE === 0) ? false : true;
export let AUTH = "";

// Helpers
export const el = (id) => document.getElementById(id);
export const fmt = (n, p=2) => (n===null || n===undefined || isNaN(Number(n))) ? "—" : Number(n).toFixed(p);
export const pct = (n) => (n===null || n===undefined) ? "—" : (n>=0? "+" : "") + fmt(n,2) + "%";
export const nowIso = () => new Date().toISOString().replace('T', ' ').slice(0,19);

// API wrapper
export async function api(path, opts={}) { /*...*/ }

```text

#### 2. **ghost-cockpit-ui.js** (~150 lines)

Button state management and UI feedback:

```javascript

import { el } from './ghost-cockpit-core.js';

export function setButtonState(btnId, state) { /*...*/ }
export function updateLatencyBadge() { /*...*/ }
export function updateAnomalyUI(flags) { /*...*/ }
export function refreshStatusBadge() { /*...*/ }

```text

#### 3. **ghost-cockpit-data.js** (~300 lines)

Data fetching and rendering:

```javascript

import { api, el, fmt, pct, nowIso } from './ghost-cockpit-core.js';
import { setButtonState } from './ghost-cockpit-ui.js';

export async function loadPortfolio() { /*...*/ }
export async function loadHeatmap() { /*...*/ }
export async function loadMovers() { /*...*/ }
export async function loadNews() { /*...*/ }
export async function loadDiagnostics() { /*...*/ }
export async function addPosition() { /*...*/ }

```text

#### 4. **ghost-cockpit-charts.js** (~200 lines)

Chart rendering logic:

```javascript

export function normalizeOverlay(d) { /*...*/ }
export function drawForecastChart(data) { /*...*/ }
export async function loadForecastOverlay() { /*...*/ }
export function connectCockpitStream() { /*...*/ }

```text

#### 5. **ghost-cockpit-stages.js** (~300 lines)

Stage 1-5 module loaders:

```javascript

export async function loadWorldContext() { /*...*/ }         // Stage 1
export async function loadAccuracyLedger() { /*...*/ }       // Stage 2
export async function runAutoTuning() { /*...*/ }            // Stage 2
export async function loadRegimeAndRisk() { /*...*/ }        // Stage 3
export async function loadPortfolioOptimization() { /*...*/ }// Stage 4
export async function loadExecutionDashboard() { /*...*/ }   // Stage 5

```text

#### 6. **ghost-cockpit-alerts.js** (~100 lines)

Corporate action banner and provider diagnostics:

```javascript

export function initCorporateActionBanner() { /*...*/ }
export function initProviderDiagnosticsPoller() { /*...*/ }

```text

#### 7. **ghost-cockpit-init.js** (~150 lines)

Main initialization orchestrator:

```javascript

import * as Core from './ghost-cockpit-core.js';
import * as UI from './ghost-cockpit-ui.js';
import * as Data from './ghost-cockpit-data.js';
import * as Charts from './ghost-cockpit-charts.js';
import * as Stages from './ghost-cockpit-stages.js';
import * as Alerts from './ghost-cockpit-alerts.js';

document.addEventListener('DOMContentLoaded', async () => {
    // Initialize auth
    await Core.initAuth();

    // Wire button events
    Core.el('btnStart').onclick = UI.startEngine;
    Core.el('btnStop').onclick = UI.stopEngine;
    // ... more wiring

    // Initial data loads
    await Data.loadPortfolio();
    await Data.loadHeatmap();
    // ... more loads

    // Start timers
    setInterval(Data.loadDiagnostics, 10000);
    setInterval(Data.loadPortfolio, 15000);
    // ... more intervals

    // Initialize alerts
    Alerts.initCorporateActionBanner();
    Alerts.initProviderDiagnosticsPoller();

    // Connect SSE streams
    Charts.connectCockpitStream();
});

```text

### Updated cockpit.html Structure

```html

<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>Ghost Protocol — Cockpit</title>
    <link rel="stylesheet" href="/static/ghost.css">
    <style>
        /*Inline critical CSS only (above-the-fold)*/
        /*Move rest to ghost-cockpit.css*/
    </style>
</head>
<body>
    <!-- Corporate Action Banner -->
    <div id="corpActionBanner" style="display:none;">...</div>

    <!-- Topbar -->
    <div class="topbar">...</div>

    <!-- Grid Cards -->
    <div class="grid">
        <!-- All section cards -->
    </div>

    <!-- Load modules -->
    <script type="module" src="/static/js/ghost-cockpit-init.js"></script>
</body>
</html>

```text

## Migration Strategy

### Phase 1: Extract Core Utilities (1 hour)

1. Create `ghost-cockpit-core.js` with helpers
2. Create `ghost-cockpit-ui.js` with button state management
3. Test in isolation with basic HTML fixture


### Phase 2: Extract Data Layer (2 hours)

1. Move all `load*` functions to `ghost-cockpit-data.js`
2. Replace inline calls with module imports
3. Test portfolio, heatmap, news, movers


### Phase 3: Extract Chart Rendering (1.5 hours)

1. Move forecast overlay logic to `ghost-cockpit-charts.js`
2. Move SSE stream handling
3. Test two-line overlay rendering


### Phase 4: Extract Stage Modules (2 hours)

1. Move Stage 1-5 loaders to `ghost-cockpit-stages.js`
2. Test world context, accuracy ledger, regime, portfolio opt, execution


### Phase 5: Extract Alert Components (1 hour)

1. Move corporate action banner to `ghost-cockpit-alerts.js`
2. Move provider diagnostics poller
3. Test banner display and dismissal


### Phase 6: Create Init Module (1 hour)

1. Create `ghost-cockpit-init.js` orchestrator
2. Wire all event handlers
3. Set up intervals and SSE connections


### Phase 7: Update cockpit.html (0.5 hours)

1. Remove all inline `<script>` blocks
2. Add single `<script type="module">` import
3. Move non-critical CSS to external file


### Phase 8: Testing & Validation (1 hour)

1. Test all features in browser
2. Verify hot reload works with --reload
3. Check browser console for errors
4. Validate SSE streams still work


**Total Estimated Time:**~10 hours

## Benefits Post-Refactor

✅**Maintainability:**Each module has clear responsibility\
✅**Testability:**Functions can be unit tested in isolation\
✅**Debuggability:**Browser DevTools show proper file names and line numbers\
✅**Reusability:**Core utilities can be shared across multiple pages\
✅**Version Control:**Clean diffs (JS changes don't mix with HTML changes)\
✅**Performance:**Browser can cache JS modules separately from HTML\
✅**Linting:**No more false positives from template literal complexity

## Backward Compatibility

- Old inline scripts can coexist during migration (but should be removed afterward)
- Module imports use `type="module"` for ES6 support (IE11 not supported, acceptable for


  internal tool)

## Testing Checklist

- [ ] Portfolio loads and updates
- [ ] Heatmap renders
- [ ] News feed populates
- [ ] Forecast overlay draws
- [ ] Corporate action banner shows/dismisses
- [ ] Provider diagnostics update
- [ ] Stage 1-5 refresh buttons work
- [ ] SSE streams connect (heartbeat + forecast_update)
- [ ] Button states animate correctly
- [ ] Latency badge updates
- [ ] Admin toggles save


## Future Enhancements

- Add TypeScript definitions for better IDE support
- Bundle modules with Rollup/Vite for production
- Add unit tests with Jest/Vitest
- Implement source maps for debugging minified code


______________________________________________________________________**Status:**📋 PLANNED - Ready for
implementation\**Priority:**Medium (quality-of-life improvement, not blocking)\**Estimated Effort:**10
hours\**Dependencies:** None (can be done incrementally)

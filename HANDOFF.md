# Ghost Protocol — Handoff Document
# Last updated: 2026-03-19 by Browser Automation Agent (Claude)

## WHAT IS THIS FILE?
This document exists so that ANY future AI agent or developer can understand
Ghost Protocol's current state without re-investigating the entire codebase.
The owner builds across many chat sessions — this prevents repeated confusion.

## SYSTEM OVERVIEW
- **What**: Autonomous AI trading system — directional predictions for crypto & stocks
- **Stack**: FastAPI + PostgreSQL + SQLite, deployed on Railway
- **Entry point**: wolf_app:APP (Procfile: web: uvicorn wolf_app:APP)
- **Production URL**: ghost-protocol-production.up.railway.app
- **Cockpit UI**: /cockpit (health monitoring dashboard)

## ARCHITECTURE (key files)
- wolf_app.py — FastAPI app, mounts all routers, starts background tasks
- wolf_helpers.py — ~590KB monolith with prediction logic, price fetching, scheduling
- engines/startup.py — Startup event handler (extracted from wolf_app Step 12)
- engines/app_config.py — Shared constants injected via globals().update()
- core/paper_tracker.py — Paper trading system (tracks virtual trades)
- core/db_pool.py — PostgreSQL connection pool (get_sync_connection is a @contextmanager)
- core/integrity.py — System health audit (run_audit returns health_score)
- routes/cockpit.py — Cockpit API endpoints (includes /integrity/audit/readonly)
- static/cockpit_v5.js — Cockpit frontend (fetches from API endpoints)
- state.py — Shared global state

## 9 INTELLIGENCE SYSTEMS
1. Ensemble Predictor — XGBoost v2, 84.8% accuracy, 59 features
2. Confidence Calibrator — Adjusts raw model confidence
3. Trust Ladder — Graduated trust levels for new symbols
4. Quality Gate — Blocks low-quality predictions
5. Prediction Killswitch — Emergency stop for bad predictions
6. VWAP Analyzer — Volume-weighted average price signals
7. World Feed Fusion — News + sentiment integration
8. Regime Detector — Market regime classification
9. Self-Improvement Engine — Learning brain, auto-correction

## CRITICAL RULES (DO NOT BREAK)
1. NEVER change prediction logic, thresholds, or model code
2. NEVER touch the database directly (use existing abstractions)
3. NEVER change the Procfile entry point (must be wolf_app:APP)
4. ALWAYS wrap risky imports in try/except with fallback
5. ALWAYS test: python -c "from wolf_app import APP; print('OK')"
6. Circular imports: wolf_helpers.py <-> engines/app_config.py need bottom injection
7. core/db_pool.py get_sync_connection() is a @contextmanager — MUST use with
8. core/paper_tracker._get_connection() is also a @contextmanager — MUST use with

## KNOWN PATTERNS
- GitHub editor auto-indents — use JavaScript CodeMirror API to bypass
- wolf_helpers.py is 590KB+ — loads slowly in GitHub editor
- engines/startup.py uses globals().update() to inject app_config constants
- Background tasks use threading.Thread (not asyncio.create_task)
- Many modules use try/except import fallbacks for resilience

## BUG TRACKER (as of 2026-03-19)

### FIXED BUGS
| Bug | Description | Commit |
|-----|-------------|--------|
| #1 | MEDIA_TEXT_HTML content type | prior |
| #2 | APP not defined | prior |
| #3 | SQL run_at column | prior |
| #4 | beast_scheduler missing | prior |
| #5 | STAGE1_ENABLED undefined | prior |
| #6 | Static mount path | prior |
| #7 | wolf_helpers injection | prior |
| #8 | startup wolf_helpers | prior |
| #9 | Circular import chain | prior |
| #10 | Prediction cycle dead | prior |
| #12 | Accuracy NO_DATA | 9748c10 |
| #13 | Financials flooding logs | prior |
| #14 | Fake health score — created /integrity/audit/readonly | 5f859d9 |
| #15 | Log level too verbose | prior |
| #20 | async event loop crash — threading.Thread | cd2f0eb |
| #21 | turbo_crypto_price error | bb3e800 |
| #22 | PRAW missing — added to requirements.txt | e13b1f5 |
| #24 | get_edge_set error | bb3e800 |
| #26 | db_engine.py IndentationError | a44e469 |
| #28 | apscheduler missing — added to requirements.txt | e13b1f5 |
| #30 | indicators.py IndentationError | 17824b3 |

### OPEN BUGS
| Bug | Description | Root Cause | Priority |
|-----|-------------|------------|----------|
| #23 | Paper trading cursor error | engines/startup.py L361,472 call _get_connection() without with | HIGH |
| #16 | Watchlist change_pct = 0 | Price change calc missing | MEDIUM |
| #25 | Pick card P&L wrong for losses | cockpit_v5.js shows gains for LOST trades | MEDIUM |
| #27 | ghost_bootstrap missing | Module never created, try/except catches it | LOW |
| #29 | Prometheus temp dir | /tmp/ghost_prom not created at startup | LOW |

### STARTUP ERRORS (non-fatal)
- api.cockpit_v2_endpoints — Module never created
- ghost_bootstrap — Module never created
- core.position_manager — Module never created
- stage2_init_failed — get_learning_loop scoping issue
- metrics_registration_failed — /tmp/ghost_prom missing

## HEALTH SCORE
- Current: ~70-80/100
- Goal: 100/100
- Check: /integrity/audit/readonly

## KEY ENDPOINTS
| Endpoint | Purpose |
|----------|---------|
| /api/health | Basic health check |
| /api/v4/picks | Current predictions |
| /api/v4/history | Prediction history |
| /api/v3/accuracy/summary | Accuracy stats |
| /api/v4/subsystems | System status |
| /integrity/audit/readonly | Full health audit (40 checks) |
| /cockpit | Web dashboard |

## FOR FUTURE AGENTS
1. Read this file FIRST before making changes
2. Check Railway deploy logs after EVERY commit (@level:error filter)
3. Split work into chunks — do not fix everything at once
4. Update this file when you fix bugs or discover new ones
5. Test endpoints after changes
6. The cockpit at /cockpit shows real-time health

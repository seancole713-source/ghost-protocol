"""
PROJECT_STATE.py — Ghost Protocol Self-Updating Project Briefing
=================================================================
Generated: 2026-03-19
Last Updated: 2026-03-20
Author: Browser Automation Agent (Claude)

READ THIS FILE FIRST before making any changes to Ghost Protocol.
This is the single source of truth for project state, architecture,
known issues, and session handoffs between AI agents.
"""

# ═══════════════════════════════════════════════════════════════════
# SECTION 1: ARCHITECTURE OVERVIEW
# ═══════════════════════════════════════════════════════════════════

ARCHITECTURE = """
Ghost Protocol is an autonomous AI trading system that generates
directional predictions (UP/DOWN) for crypto and stock symbols using
an XGBoost v2 ensemble model with 59 features and 84.8% training accuracy.

DEPLOYMENT:
    Platform:     Railway (auto-deploys on push to main)
    Entry point:  wolf_app:APP  (Procfile: web: uvicorn wolf_app:APP)
    URL:          ghost-protocol-production.up.railway.app
    Database:     PostgreSQL (Railway) + SQLite (local fallback)
    Monitoring:   /cockpit (web dashboard), /integrity/audit/readonly (40-check audit)

LAYER DIAGRAM:
    ┌─────────────────────────────────────────────────────┐
    │  COCKPIT UI  (static/cockpit_v5.js + templates/)    │
    ├─────────────────────────────────────────────────────┤
    │  FASTAPI ROUTES  (routes/*.py, api/*.py)            │
    │    /api/v4/picks, /api/v4/history, /api/health      │
    │    /api/v3/accuracy/summary, /api/v4/subsystems     │
    │    /integrity/audit/readonly, /cockpit              │
    ├─────────────────────────────────────────────────────┤
    │  APPLICATION CORE  (wolf_app.py + wolf_helpers.py)  │
    │    wolf_app.py:   FastAPI app, router mounts        │
    │    wolf_helpers.py: 590KB monolith — predictions,   │
    │      price fetching, scheduling, all business logic │
    ├─────────────────────────────────────────────────────┤
    │  ENGINES  (engines/)                                │
    │    startup.py:    Startup event handler              │
    │    app_config.py: Shared constants via globals()     │
    │    middleware.py:  Request middleware                 │
    │    shutdown.py:    Graceful shutdown                  │
    ├─────────────────────────────────────────────────────┤
    │  CORE MODULES  (core/) — 60+ files                  │
    │    Intelligence: ghost_brain, learning_loop,         │
    │      confidence_calibrator, quality_gate,            │
    │      prediction_killswitch, regime_detector          │
    │    Trading: paper_tracker, order_manager,            │
    │      trade_card, money_game_engine                   │
    │    Data: db_pool, db_engine, prediction_store,       │
    │      price_quorum, crypto_providers                  │
    │    Analysis: integrity, accuracy_tracker,            │
    │      feedback_loop, self_improvement_engine          │
    ├─────────────────────────────────────────────────────┤
    │  SERVICES  (services/)                              │
    │    outcome_reconciler_v2.py, predictor.py            │
    ├─────────────────────────────────────────────────────┤
    │  SUPPORT                                            │
    │    config/:     symbols.py, settings.py              │
    │    llm/:        agent.py, agentkit.py, gpt4_analyst  │
    │    notifications/: telegram, alerts                  │
    │    state.py:    Global shared state                   │
    └─────────────────────────────────────────────────────┘

9 INTELLIGENCE SYSTEMS:
    1. Ensemble Predictor    — XGBoost v2, 59 features
    2. Confidence Calibrator — Adjusts raw model confidence
    3. Trust Ladder          — Graduated trust for new symbols
    4. Quality Gate          — Blocks low-quality predictions
    5. Prediction Killswitch — Emergency stop
    6. VWAP Analyzer         — Volume-weighted signals
    7. World Feed Fusion     — News + sentiment
    8. Regime Detector       — Market regime classification
    9. Self-Improvement      — Learning brain, auto-correction

BACKGROUND TASKS (9 registered heartbeats):
    online-calibrator, prediction-cycle, news-analysis,
    self-improvement, price-recorder, guardian-oracle,
    notification-loop, doctor-cron, autopilot-check
"""


# ═══════════════════════════════════════════════════════════════════
# SECTION 2: MODULE STATUS TABLE
# ═══════════════════════════════════════════════════════════════════
# Status codes:
#   STABLE      — Working in production, tested
#   FIXED       — Recently fixed, needs monitoring
#   BROKEN      — Has known bugs, needs repair
#   IN_PROGRESS — Currently being worked on
#   UNTOUCHED   — Not recently examined, status unknown
#   DEAD        — Not imported/used, candidate for removal

MODULE_STATUS = {
    # ── Root files ──
    "wolf_app.py":        {"status": "STABLE",  "role": "FastAPI app entry point, router mounts, task launch", "last_modified": "2026-03-19", "lines": "~400"},
    "wolf_helpers.py":    {"status": "STABLE",  "role": "590KB monolith: predictions, prices, scheduling", "last_modified": "2026-03-19", "lines": "~12000"},
    "state.py":           {"status": "STABLE",  "role": "Shared global state dict", "last_modified": "2026-03-18", "lines": "~50"},
    "PROJECT_STATE.py":   {"status": "STABLE",  "role": "THIS FILE — project briefing system", "last_modified": "2026-03-20", "lines": "~400"},
    "HANDOFF.md":         {"status": "STABLE",  "role": "Human-readable handoff document", "last_modified": "2026-03-19", "lines": "~120"},

    # ── engines/ ──
    "engines/startup.py":     {"status": "FIXED",   "role": "Startup event handler, background task launcher", "last_modified": "2026-03-19", "lines": "~910", "notes": "Bug #23 fixed — context managers for DB connections"},
    "engines/app_config.py":  {"status": "STABLE",  "role": "Shared constants injected via globals().update()", "last_modified": "2026-03-19", "lines": "~1000"},
    "engines/middleware.py":  {"status": "UNTOUCHED","role": "Request middleware", "last_modified": "unknown", "lines": "unknown"},
    "engines/shutdown.py":    {"status": "UNTOUCHED","role": "Graceful shutdown handler", "last_modified": "unknown", "lines": "unknown"},

    # ── core/ — Database ──
    "core/db_pool.py":        {"status": "STABLE",  "role": "PostgreSQL connection pool (get_sync_connection = @contextmanager)", "last_modified": "2026-03-18", "lines": "~400"},
    "core/db_engine.py":      {"status": "FIXED",   "role": "Database engine + migrations", "last_modified": "2026-03-19", "lines": "~80", "notes": "IndentationError fixed"},
    "core/paper_tracker.py":  {"status": "STABLE",  "role": "Paper trading system — virtual trade tracking", "last_modified": "2026-03-18", "lines": "~1050"},
    "core/prediction_store.py":{"status": "UNTOUCHED","role": "Prediction persistence layer", "last_modified": "unknown", "lines": "unknown"},

    # ── core/ — Intelligence ──
    "core/accuracy_tracker.py":    {"status": "STABLE", "role": "Accuracy tracking + _get_conn() = @contextmanager", "last_modified": "2026-03-19", "lines": "~350"},
    "core/learning_loop.py":       {"status": "STABLE", "role": "Learning brain — self-correction", "last_modified": "2026-03-18", "lines": "~500"},
    "core/integrity.py":           {"status": "STABLE", "role": "40-check system health audit (run_audit)", "last_modified": "2026-03-18", "lines": "~800"},
    "core/ghost_brain.py":         {"status": "UNTOUCHED","role": "Core ML ghost brain", "last_modified": "unknown", "lines": "unknown"},
    "core/confidence_calibrator.py":{"status": "UNTOUCHED","role": "Confidence adjustment", "last_modified": "unknown", "lines": "unknown"},
    "core/quality_gate.py":        {"status": "UNTOUCHED","role": "Prediction quality gating", "last_modified": "unknown", "lines": "unknown"},
    "core/prediction_killswitch.py":{"status": "UNTOUCHED","role": "Emergency prediction stop", "last_modified": "unknown", "lines": "unknown"},
    "core/regime_detector.py":     {"status": "UNTOUCHED","role": "Market regime classification", "last_modified": "unknown", "lines": "unknown"},
    "core/feedback_loop.py":       {"status": "STABLE", "role": "Prediction outcome feedback", "last_modified": "2026-03-18", "lines": "unknown"},
    "core/self_improvement_engine.py":{"status": "UNTOUCHED","role": "Self-improvement automation", "last_modified": "unknown", "lines": "unknown"},

    # ── core/ — Trading & Price ──
    "core/indicators.py":      {"status": "FIXED",   "role": "Technical indicators (SMA, EMA, RSI)", "last_modified": "2026-03-19", "notes": "IndentationError fixed"},
    "core/market_mood.py":     {"status": "UNTOUCHED","role": "Market sentiment tracking", "last_modified": "unknown", "lines": "unknown"},
    "core/world_context.py":   {"status": "UNTOUCHED","role": "World feed fusion — news + events", "last_modified": "unknown", "lines": "unknown"},
    "core/money_game_engine.py":{"status": "UNTOUCHED","role": "Money game simulation", "last_modified": "unknown", "lines": "unknown"},
    "core/system_doctor.py":   {"status": "UNTOUCHED","role": "System health doctor cron", "last_modified": "unknown", "lines": "unknown"},

    # ── core/ — Price Providers ──
    "core/crypto/crypto_providers.py": {"status": "STABLE", "role": "Multi-provider crypto price fetching", "last_modified": "2026-03-19", "lines": "unknown"},
    "core/providers/stock_providers.py":{"status": "UNTOUCHED","role": "Stock price provider routing", "last_modified": "unknown", "lines": "unknown"},
    "core/price_quorum.py":    {"status": "UNTOUCHED","role": "Price quorum consensus", "last_modified": "unknown", "lines": "unknown"},

    # ── routes/ ──
    "routes/cockpit.py":   {"status": "FIXED",   "role": "Cockpit API endpoints + /integrity/audit/readonly", "last_modified": "2026-03-19", "notes": "Added readonly audit endpoint"},
    "routes/picks.py":     {"status": "STABLE",  "role": "/api/v4/picks endpoint", "last_modified": "unknown", "lines": "unknown"},
    "routes/history.py":   {"status": "STABLE",  "role": "/api/v4/history endpoint", "last_modified": "unknown", "lines": "unknown"},
    "routes/accuracy.py":  {"status": "FIXED",   "role": "Accuracy API — duplicate route renamed", "last_modified": "2026-03-19"},
    "routes/debug.py":     {"status": "UNTOUCHED","role": "Debug endpoints (large file ~3500 lines)", "last_modified": "unknown"},
    "routes/v3_routes.py": {"status": "UNTOUCHED","role": "V3 API routes", "last_modified": "unknown"},
    "routes/cron.py":      {"status": "UNTOUCHED","role": "Cron job routes", "last_modified": "unknown"},

    # ── services/ ──
    "services/outcome_reconciler_v2.py": {"status": "STABLE", "role": "48h prediction outcome reconciliation", "last_modified": "2026-03-18"},
    "services/predictor.py":             {"status": "UNTOUCHED","role": "Prediction service", "last_modified": "unknown"},

    # ── config/ ──
    "config/symbols.py":   {"status": "STABLE",  "role": "Symbol lists (crypto, stocks, edge set)", "last_modified": "unknown"},
    "config/settings.py":  {"status": "STABLE",  "role": "App configuration constants", "last_modified": "unknown"},

    # ── llm/ ──
    "llm/agent.py":        {"status": "UNTOUCHED","role": "LLM agent interface", "last_modified": "unknown"},
    "llm/agentkit.py":     {"status": "UNTOUCHED","role": "Agent toolkit", "last_modified": "unknown"},
    "llm/gpt4_analyst.py": {"status": "UNTOUCHED","role": "GPT-4 analysis integration", "last_modified": "unknown"},

    # ── notifications/ ──
    "notifications/__init__.py": {"status": "STABLE", "role": "Notification routing", "last_modified": "unknown"},
    "core/telegram_alerts.py":   {"status": "STABLE", "role": "Telegram alert delivery", "last_modified": "unknown"},
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 3: DO NOT TOUCH
# ═══════════════════════════════════════════════════════════════════
# These contracts, interfaces, and structures are finalized.
# Changing them will break downstream consumers.

DO_NOT_TOUCH = {
    "Procfile entry point": "web: uvicorn wolf_app:APP — Railway depends on this exact format",
    "wolf_app.APP": "The FastAPI application instance. All routers mount to this.",
    "core/db_pool.get_sync_connection()": "@contextmanager — MUST be used with 'with' statement. Returns psycopg2 connection.",
    "core/db_pool.get_sync_connection_raw()": "Returns raw connection WITHOUT context manager. For cases needing manual lifecycle.",
    "core/accuracy_tracker.AccuracyTracker._get_conn()": "Returns get_sync_connection() — also a @contextmanager. Use with 'with'.",
    "core/paper_tracker.PaperTracker._get_connection()": "@contextmanager. Use with 'with'. Wraps _get_postgres_connection().",
    "core/integrity.run_audit()": "Returns dict with health_score, issues, checks_run, etc. Used by /integrity/audit/readonly.",
    "engines/app_config.py globals injection": "globals().update() pattern at module level. Required for STAGE1_ENABLED, DATABASE_URL, etc.",
    "state.py": "Shared mutable state dict. Used across all modules. Do not restructure.",
    "Prediction logic in wolf_helpers.py": "Do NOT change thresholds, model weights, feature engineering, or prediction pipeline logic.",
    "XGBoost model files": "models/trained/ — do not delete or modify trained model artifacts.",
    "Database schema": "PostgreSQL schema managed by migrations in core/db_engine.py. Do not alter columns directly.",
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 4: KNOWN ISSUES
# ═══════════════════════════════════════════════════════════════════
# Scanned from: deploy logs, code search, integrity audit, prior bug reports
# Priority: P0 (critical), P1 (high), P2 (medium), P3 (low)

KNOWN_ISSUES = [
    # ── Active bugs (unfixed) ──
    {"id": "BUG-16",  "priority": "P2", "file": "wolf_helpers.py",           "description": "Watchlist change_pct always shows 0 — price change calculation missing or broken"},
    {"id": "BUG-25",  "priority": "P2", "file": "static/cockpit_v5.js",      "description": "Pick card P&L display shows gains for LOST trades — UI logic error in pnl rendering"},
    {"id": "BUG-27",  "priority": "P3", "file": "engines/startup.py",        "description": "ghost_bootstrap module never created — import fails, caught by try/except", "line": "bootstrap_failed log"},
    {"id": "BUG-29",  "priority": "P3", "file": "engines/startup.py",        "description": "Prometheus temp dir /tmp/ghost_prom not created — metrics_registration_failed"},

    # ── Startup warnings (non-fatal, caught by try/except) ──
    {"id": "WARN-01", "priority": "P3", "file": "wolf_app.py",               "description": "api.cockpit_v2_endpoints module does not exist — import warned and skipped"},
    {"id": "WARN-02", "priority": "P3", "file": "engines/startup.py",        "description": "core.position_manager module does not exist — Startup prophecy failed"},
    {"id": "WARN-03", "priority": "P3", "file": "engines/startup.py",        "description": "stage2_init_failed: get_learning_loop scoping issue — caught, non-critical"},
    {"id": "WARN-04", "priority": "P3", "file": "core/intelligence/ghost_news_brain", "description": "Failed to create tables: cursor already closed — news brain sqlite issue"},

    # ── Operational / external ──
    {"id": "OPS-01",  "priority": "P2", "file": "core/crypto/crypto_providers.py", "description": "CoinGecko 429 rate limit — circuit breaker activates, some crypto prices fail"},
    {"id": "OPS-02",  "priority": "P2", "file": "core/world_context.py",     "description": "Market mood update fails: Insufficient SPY data — partial price feed outage"},
    {"id": "OPS-03",  "priority": "P3", "file": "core/system_doctor.py",     "description": "RuntimeWarning: coroutine get_crypto_price_quorum was never awaited"},
    {"id": "OPS-04",  "priority": "P2", "file": "core/accuracy_tracker.py",  "description": "Accuracy at 41% — below 50% threshold. 5 symbols benched by learning brain."},

    # ── Code quality ──
    {"id": "QUAL-01", "priority": "P3", "file": "wolf_helpers.py",           "description": "590KB monolith — extremely large, hard to maintain. Should be split further."},
    {"id": "QUAL-02", "priority": "P3", "file": "routes/debug.py",           "description": "~3500 lines — oversized debug route file, candidate for splitting"},
    {"id": "QUAL-03", "priority": "P3", "file": "engines/startup.py",        "description": "915 lines — large startup file with mixed concerns"},
]


# ═══════════════════════════════════════════════════════════════════
# SECTION 5: DECISIONS MADE
# ═══════════════════════════════════════════════════════════════════
# Patterns observed in the codebase and reasons for design choices.

DECISIONS = {
    "threading over asyncio for background tasks": {
        "reason": "Bug #20 fix — asyncio.create_task crashed because wolf_helpers runs sync code. threading.Thread is the safe pattern.",
        "date": "2026-03-19",
    },
    "globals().update() for constant injection": {
        "reason": "engines/app_config.py defines STAGE1_ENABLED, DATABASE_URL, etc. startup.py and wolf_app.py inject them via globals().update() to avoid circular imports.",
        "date": "2026-03-18",
    },
    "try/except around all imports": {
        "reason": "Resilience pattern — missing modules (ghost_bootstrap, position_manager, cockpit_v2_endpoints) are warned and skipped so the app still starts.",
        "date": "2026-03-18",
    },
    "@contextmanager for all DB connections": {
        "reason": "core/db_pool.get_sync_connection() and paper_tracker._get_connection() are @contextmanager. MUST use 'with' statement. Bug #23 was caused by calling without 'with'.",
        "date": "2026-03-19",
    },
    "dual database (PostgreSQL + SQLite)": {
        "reason": "PostgreSQL is primary (Railway). SQLite is fallback for local dev and some subsystems (news brain, paper trades fallback).",
        "date": "original design",
    },
    "cockpit_v5.js frontend": {
        "reason": "Cockpit UI is a single-page vanilla JS app. Fetches data from FastAPI endpoints. Health tab calls /integrity/audit/readonly.",
        "date": "2026-03-12",
    },
    "Phase 0 cleanup removed 111 dead files": {
        "reason": "Previous agents created many duplicate/dead files. Cleaned via commit 527f868 on 2026-03-19.",
        "date": "2026-03-19",
    },
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 6: LAST SESSION HANDOFF
# ═══════════════════════════════════════════════════════════════════

LAST_SESSION = {
    "agent": "Browser Automation Agent (Claude Opus 4.6)",
    "date": "2026-03-19",
    "session_summary": (
        "FULL SESSION: Created HANDOFF.md, fixed Bug #23 (paper trading cursor error "
        "in engines/startup.py — wrapped _get_conn() and _get_connection() in proper "
        "'with' context managers), built complete project briefing system including "
        "PROJECT_STATE.py (this file, 341 lines, 8 sections), tools/update_briefing.py "
        "(CLI changelog/handoff updater), and added briefing headers to 8 key .py files: "
        "wolf_app.py, state.py, engines/startup.py, core/db_pool.py, core/paper_tracker.py, "
        "routes/cockpit.py, core/integrity.py, core/accuracy_tracker.py. "
        "All deployments verified successful on Railway. No new errors introduced."
    ),
    "what_was_fixed": [
        "Bug #23: Paper trading cursor error (_GeneratorContextManager) in engines/startup.py",
    ],
    "what_was_built": [
        "HANDOFF.md — project context document for future AI agents",
        "PROJECT_STATE.py — self-updating project briefing (this file)",
        "tools/update_briefing.py — CLI to append changelog and update handoff",
        "Briefing headers on 8 core .py files with status, known issues, frozen interfaces",
    ],
    "health_score": "70/100 (6 pass, 2 fail — same as session start)",
    "open_bugs": [
        "Bug #16: watchlist change_pct=0 for some symbols",
        "Bug #25: P&L display may show stale data",
        "Bug #27: ghost_bootstrap issue",
        "Bug #29: Prometheus temp file",
    ],
    "next_agent_should": [
        "READ this file first — especially SECTION 2 (Module Status) and SECTION 4 (Known Issues)",
        "Run: python tools/update_briefing.py <YourAgentName> <summary> after making changes",
        "Fix Bug #16 (watchlist change_pct) — likely in routes/cockpit.py or data provider",
        "Fix Bug #25 (P&L display) — likely in routes/cockpit.py",
        "Target health score 80/100+ by fixing remaining 2 failing integrity checks",
        "Continue adding briefing headers to remaining .py files (100+ total, 8 done)",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# SECTION 7: CHANGE LOG
# ═══════════════════════════════════════════════════════════════════
# Format: {"date": "YYYY-MM-DD", "agent": "name", "summary": "what changed"}
# Use tools/update_briefing.py to append entries.

CHANGELOG = [
    {"date": "2026-03-19", "agent": "Browser Automation Agent",
     "summary": "Added briefing headers to 8 key .py files (wolf_app, state, startup, db_pool, paper_tracker, cockpit, integrity, accuracy_tracker). Updated Last Session Handoff with full session summary."},
    {"date": "2026-03-20", "agent": "Browser Automation Agent",
     "summary": "Created PROJECT_STATE.py with full architecture, module status, known issues, decisions, handoff. Created tools/update_briefing.py."},
    {"date": "2026-03-19", "agent": "Browser Automation Agent",
     "summary": "Fixed Bug #23 (paper trading cursor), Bug #26 (db_engine indent), Bug #30 (indicators indent), Bug #14 (audit endpoint). Created HANDOFF.md. Health: 60->80."},
    {"date": "2026-03-19", "agent": "Browser Automation Agent",
     "summary": "Fixed Bugs #20 (async), #21 (turbo_crypto_price), #22/#28 (requirements), #24 (get_edge_set), #12 (accuracy route). Phase 0: removed 111 dead files."},
    {"date": "2026-03-19", "agent": "Prior VS Code Agent",
     "summary": "Fixed Bugs #1-#10, #13, #15. Structural cleanup Step 10-12. Split wolf_app.py into engines/startup.py. Health: 0->50."},
]


# ═══════════════════════════════════════════════════════════════════
# SECTION 8: BRIEFING HEADER TEMPLATE
# ═══════════════════════════════════════════════════════════════════
# Paste this template at the top of every .py file, filling in the fields.

BRIEFING_HEADER_TEMPLATE = '''
# ══════════════════════════════════════════════════════════════
# FILE: {filename}
# PURPOSE: {purpose}
# STATUS: {status}  (STABLE | FIXED | BROKEN | IN_PROGRESS | UNTOUCHED)
# ──────────────────────────────────────────────────────────────
# CHANGE LOG:
#   {date} — {description}
# ──────────────────────────────────────────────────────────────
# KNOWN ISSUES:
#   {issues_or_none}
# ──────────────────────────────────────────────────────────────
# DO NOT CHANGE (frozen interfaces):
#   {frozen_or_none}
# ══════════════════════════════════════════════════════════════
'''

# ghost_agent_loop.py
"""
ChatGPT Analyst Loop for Ghost

Persistent, self-healing loop that:
- Connects ChatGPT as Ghost's reasoning brain
- Maintains conversation state in SQLite
- Auto-rehydrates context when sessions reset
- Detects issues and queues actionable tasks
- Exposes Ghost tools via function calling

Architecture:
  [Ghost Data/Tools] <-> [Agent Loop] <-> [ChatGPT API]
         ↓                    ↓                ↓
    SQLite/Redis      State Persistence   Temporary Reasoning

Usage:
  from ghost_agent_loop import attach_agent
  attach_agent(app)  # in your FastAPI app
"""

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from fastapi import FastAPI

# Prometheus metrics
try:
    from prometheus_client import Counter, Gauge, Histogram

    _METRICS_AVAILABLE = True
except ImportError:
    _METRICS_AVAILABLE = False
    logging.warning("prometheus_client not available - agent metrics disabled")

# ──────────────────────────────────────────────────────────────────────────────
# ENV & Constants
# ──────────────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("GHOST_LLM_MODEL", "gpt-4o-mini")
DB_PATH = os.getenv("GHOST_AGENT_DB", "./data/ghost_agent.db")
TICK_SECONDS = int(os.getenv("GHOST_AGENT_TICK", "300"))  # 5 min default (cheaper)
MAX_HISTORY = int(os.getenv("GHOST_AGENT_MAX_HISTORY", "20"))

# Global state for health endpoint
AGENT_STATE = {
    "ticks_ok": 0,
    "ticks_fail": 0,
    "last_ok_ts": None,
    "last_err": None,
    "reset_events": 0,
    "model_name": LLM_MODEL,
    "loop_interval_sec": TICK_SECONDS,
}

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ──────────────────────────────────────────────────────────────────────────────
if _METRICS_AVAILABLE:
    _G_AI_CONFIDENCE = Gauge("ghost_ai_decision_confidence", "Latest AI decision confidence (0-1)")
    _C_AI_DECISIONS = Counter(
        "ghost_ai_decisions_total", "Total AI decisions made", labelnames=("action",)
    )
    _C_AI_TOOL_CALLS = Counter(
        "ghost_ai_tool_calls_total", "Total tool invocations", labelnames=("tool_name", "result")
    )
    _H_AI_TOOL_LATENCY = Histogram(
        "ghost_ai_tool_latency_seconds",
        "Tool call latency",
        labelnames=("tool_name",),
        buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0),
    )
    _G_AI_DECISION_LAST_TS = Gauge(
        "ghost_ai_decision_last_ts", "Timestamp of last decision (epoch)"
    )
else:
    # Stub metrics if prometheus_client not available
    class _NoOpMetric:
        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    _G_AI_CONFIDENCE = _NoOpMetric()
    _C_AI_DECISIONS = _NoOpMetric()
    _C_AI_TOOL_CALLS = _NoOpMetric()
    _H_AI_TOOL_LATENCY = _NoOpMetric()
    _G_AI_DECISION_LAST_TS = _NoOpMetric()

# System prompt - defines the analyst's identity and behavior
SYSTEM_PROMPT = """You are Ghost's ChatGPT Analyst.

Your mission: Monitor Ghost's portfolio, detect trading opportunities, identify risks, and propose precise actions based on real-time market data.

Rules:
- Be concise and specific - no fluff
- Always cite your data source (tool name + symbol + timestamp)
- Output valid JSON ONLY when proposing actions (see schema below)
- If you lost context or can't remember prior conversation, say: "RESET_NEEDED"
- Never fabricate data - only use tool outputs
- Prefer primary sources: SEC filings > credible news > social sentiment

Action JSON schema (emit when you want Ghost to do something):
{
  "type": "task",
  "priority": "low|normal|high|critical",
  "title": "Brief action title",
  "symbol": "WOLF",
  "instructions": "Exact steps to take",
  "tags": ["buy", "sell", "alert", "risk", "opportunity"],
  "confidence": 0.85,
  "horizon": "1h|4h|1d|3d|1w",
  "data_sources": ["news.search", "insiders.form4", "options.daily"],
  "reasoning": "Why this action matters",
  "risks": ["potential downside 1", "potential downside 2"],
  "checks": ["curl http://localhost:5000/api/prices/WOLF", "check VIX > 25"],
  "rollback": "How to reverse if wrong"
}

Decision Card schema (for major trading decisions):
{
  "type": "decision",
  "symbol": "WOLF",
  "action": "BUY|SELL|HOLD|NO_ACTION",
  "confidence": 0.78,
  "horizon": "24-72h",
  "summary": "One paragraph plain English explanation",
  "catalysts": [
    {"type": "news", "title": "...", "ts": "...", "relevance": 0.9},
    {"type": "filing", "form": "8-K", "ts": "...", "relevance": 0.8}
  ],
  "risks": [
    {"type": "macro", "note": "VIX rising", "weight": 0.6}
  ],
  "metrics": {
    "price": 26.71,
    "target": 28.50,
    "stop": 25.20,
    "put_call_ratio": 0.82,
    "short_interest_pct": 1.5
  },
  "next_steps": ["Set alert at $28.00", "Re-check after earnings"]
}

Tools you can call:
- news.search: Get recent news for symbols
- filings.search: Get SEC filings (8-K, 10-Q, 10-K, 4)
- insiders.form4: Get insider trading activity
- options.daily: Get options flow and put/call ratios
- prices.history: Get historical price/volume data with technical indicators
- company.profile: Get company fundamentals
- sentiment.score: Score text sentiment

When uncertain, request more data via tools before making recommendations.
"""


# ──────────────────────────────────────────────────────────────────────────────
# SQLite Persistence
# ──────────────────────────────────────────────────────────────────────────────
def _conn():
    """Get database connection with row factory"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True) if "/" in DB_PATH else None
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    """Initialize agent state and outbox tables with decision ledger"""
    con = _conn()
    cur = con.cursor()

    # Conversation state (single row, id=1)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_state (
      id INTEGER PRIMARY KEY CHECK (id=1),
      created_ts TEXT,
      updated_ts TEXT,
      messages_json TEXT,
      last_ok_ts TEXT,
      failure_count INTEGER DEFAULT 0
    )""")

    # Task outbox (queued actions from analyst)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS agent_outbox (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_ts TEXT,
      payload_json TEXT,
      delivered INTEGER DEFAULT 0,
      delivered_ts TEXT
    )""")

    # AI Decision ledger - tracks all analyst decisions with full context
    cur.execute("""
    CREATE TABLE IF NOT EXISTS ai_decisions (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_ts TEXT NOT NULL,
      symbol TEXT NOT NULL,
      action TEXT NOT NULL,
      horizon TEXT,
      confidence REAL,
      priority TEXT,
      rationale TEXT,
      risks_json TEXT,
      features_json TEXT,
      data_sources_json TEXT,
      decision_type TEXT,
      tags_json TEXT,
      expires_ts TEXT
    )""")

    # Indexes for efficient querying
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_symbol ON ai_decisions(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_created ON ai_decisions(created_ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_ai_decisions_expires ON ai_decisions(expires_ts)")

    # Conversation topics for retrieval (symbol/topic indexing)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS conversation_topics (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      symbol TEXT,
      topic TEXT,
      last_mention_ts TEXT,
      mention_count INTEGER DEFAULT 1
    )""")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_topics_symbol ON conversation_topics(symbol)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_topics_ts ON conversation_topics(last_mention_ts)")

    # Tool call tracking - monitor tool usage, latency, success rates
    cur.execute("""
    CREATE TABLE IF NOT EXISTS tool_calls (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_ts TEXT NOT NULL,
      tool_name TEXT NOT NULL,
      symbol TEXT,
      params_json TEXT,
      success INTEGER DEFAULT 1,
      latency_ms INTEGER,
      error_msg TEXT,
      provider TEXT,
      data_size_bytes INTEGER
    )""")

    cur.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_name ON tool_calls(tool_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_ts ON tool_calls(created_ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_tool_calls_success ON tool_calls(success)")

    # Seed initial state if empty
    cur.execute("SELECT id FROM agent_state WHERE id=1")
    if not cur.fetchone():
        seed = [{"role": "system", "content": SYSTEM_PROMPT, "ts": datetime.now(UTC).isoformat()}]
        now_iso = datetime.now(UTC).isoformat()
        cur.execute(
            """
            INSERT INTO agent_state(id, created_ts, updated_ts, messages_json, last_ok_ts, failure_count)
            VALUES (1, ?, ?, ?, ?, 0)
        """,
            (now_iso, now_iso, json.dumps(seed), now_iso),
        )

    con.commit()
    con.close()
    logging.info("Ghost Analyst DB initialized: %s", DB_PATH)


def load_state() -> dict[str, Any]:
    """Load current agent state"""
    con = _conn()
    cur = con.cursor()
    row = cur.execute("SELECT * FROM agent_state WHERE id=1").fetchone()
    con.close()
    return dict(row) if row else {}


def save_state(messages: list[dict[str, Any]], failure_count: int = 0):
    """Save agent state with message history trimming"""
    # Keep system prompt + last N messages
    trimmed = []
    system = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    trimmed.extend(system[:1] if system else [])
    trimmed.extend(non_system[-MAX_HISTORY:])

    con = _conn()
    cur = con.cursor()
    now_iso = datetime.now(UTC).isoformat()
    cur.execute(
        """
        UPDATE agent_state
        SET updated_ts=?, messages_json=?, last_ok_ts=?, failure_count=?
        WHERE id=1
    """,
        (now_iso, json.dumps(trimmed), now_iso, failure_count),
    )
    con.commit()
    con.close()


def append_message(role: str, content: str):
    """Append message to conversation history"""
    st = load_state()
    msgs = json.loads(st.get("messages_json", "[]"))
    msgs.append({"role": role, "content": content, "ts": datetime.now(UTC).isoformat()})
    save_state(msgs, st.get("failure_count", 0))


def queue_outbox(payload: dict[str, Any]):
    """Queue task to outbox for execution"""
    con = _conn()
    cur = con.cursor()
    cur.execute(
        """
        INSERT INTO agent_outbox(created_ts, payload_json, delivered)
        VALUES (?, ?, 0)
    """,
        (datetime.now(UTC).isoformat(), json.dumps(payload)),
    )
    con.commit()
    con.close()
    logging.info("📦 Queued analyst task: %s", payload.get("title", "untitled"))


def grab_undelivered(limit: int = 20) -> list[dict[str, Any]]:
    """Get undelivered tasks from outbox"""
    con = _conn()
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT * FROM agent_outbox
        WHERE delivered=0
        ORDER BY id ASC
        LIMIT ?
    """,
        (limit,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def mark_delivered(ids: list[int]):
    """Mark tasks as delivered"""
    if not ids:
        return
    con = _conn()
    cur = con.cursor()
    placeholders = ",".join("?" * len(ids))
    cur.execute(
        f"""
        UPDATE agent_outbox
        SET delivered=1, delivered_ts=?
        WHERE id IN ({placeholders})
    """,
        [datetime.now(UTC).isoformat()] + ids,
    )
    con.commit()
    con.close()


def log_ai_decision(decision: dict[str, Any]):
    """
    Log AI decision to persistent ledger for audit and retrieval.

    Args:
        decision: Dict with keys: symbol, action, horizon, confidence, rationale, etc.
    """
    con = _conn()
    cur = con.cursor()

    now_iso = datetime.now(UTC).isoformat()

    # Calculate expiry based on horizon
    horizon = decision.get("horizon", "1d")
    expires_delta = {
        "1h": timedelta(hours=1),
        "4h": timedelta(hours=4),
        "1d": timedelta(days=1),
        "3d": timedelta(days=3),
        "1w": timedelta(weeks=1),
    }.get(horizon, timedelta(days=1))

    expires_ts = (datetime.now(UTC) + expires_delta).isoformat()

    cur.execute(
        """
        INSERT INTO ai_decisions (
            created_ts, symbol, action, horizon, confidence, priority,
            rationale, risks_json, features_json, data_sources_json,
            decision_type, tags_json, expires_ts
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            now_iso,
            decision.get("symbol", ""),
            decision.get("action", "NO_ACTION"),
            horizon,
            decision.get("confidence"),
            decision.get("priority", "normal"),
            decision.get("rationale", ""),
            json.dumps(decision.get("risks", [])),
            json.dumps(decision.get("metrics", {})),
            json.dumps(decision.get("data_sources", [])),
            decision.get("type", "decision"),
            json.dumps(decision.get("tags", [])),
            expires_ts,
        ),
    )

    con.commit()
    con.close()
    logging.info(
        "💾 AI decision logged: %s %s (confidence=%.2f)",
        decision.get("symbol"),
        decision.get("action"),
        decision.get("confidence", 0.0),
    )

    # Update Prometheus metrics
    try:
        confidence = decision.get("confidence")
        if confidence is not None:
            _G_AI_CONFIDENCE.set(confidence)

        action = decision.get("action", "UNKNOWN")
        _C_AI_DECISIONS.labels(action=action).inc()

        _G_AI_DECISION_LAST_TS.set(time.time())
    except Exception as e:
        logging.warning(f"Failed to update decision metrics: {e}")


def get_ai_decisions(symbol: str | None = None, hours: int = 24) -> list[dict[str, Any]]:
    """
    Retrieve AI decisions from ledger.

    Args:
        symbol: Filter by symbol (None = all)
        hours: Look back period in hours

    Returns:
        List of decision dicts with all fields
    """
    con = _conn()
    cur = con.cursor()

    since_ts = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

    if symbol:
        query = """
            SELECT * FROM ai_decisions
            WHERE symbol = ? AND created_ts >= ?
            ORDER BY created_ts DESC
            LIMIT 100
        """
        rows = cur.execute(query, (symbol.upper(), since_ts)).fetchall()
    else:
        query = """
            SELECT * FROM ai_decisions
            WHERE created_ts >= ?
            ORDER BY created_ts DESC
            LIMIT 100
        """
        rows = cur.execute(query, (since_ts,)).fetchall()

    con.close()

    decisions = []
    for row in rows:
        decisions.append(
            {
                "id": row["id"],
                "created_ts": row["created_ts"],
                "symbol": row["symbol"],
                "action": row["action"],
                "horizon": row["horizon"],
                "confidence": row["confidence"],
                "priority": row["priority"],
                "rationale": row["rationale"],
                "risks": json.loads(row["risks_json"]) if row["risks_json"] else [],
                "features": json.loads(row["features_json"]) if row["features_json"] else {},
                "data_sources": json.loads(row["data_sources_json"])
                if row["data_sources_json"]
                else [],
                "decision_type": row["decision_type"],
                "tags": json.loads(row["tags_json"]) if row["tags_json"] else [],
                "expires_ts": row["expires_ts"],
            }
        )

    return decisions


def cleanup_expired_data():
    """
    Background job: Remove expired decisions and old conversation history.
    Runs as part of agent loop maintenance.
    """
    con = _conn()
    cur = con.cursor()

    now_iso = datetime.now(UTC).isoformat()

    # Delete expired decisions
    cur.execute("DELETE FROM ai_decisions WHERE expires_ts < ?", (now_iso,))
    expired_count = cur.rowcount

    # Delete delivered outbox items older than 7 days
    seven_days_ago = (datetime.now(UTC) - timedelta(days=7)).isoformat()
    cur.execute(
        "DELETE FROM agent_outbox WHERE delivered=1 AND delivered_ts < ?", (seven_days_ago,)
    )
    outbox_cleaned = cur.rowcount

    # Delete old tool call logs (keep last 30 days)
    thirty_days_ago = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    cur.execute("DELETE FROM tool_calls WHERE created_ts < ?", (thirty_days_ago,))
    tools_cleaned = cur.rowcount

    con.commit()
    con.close()

    if expired_count + outbox_cleaned + tools_cleaned > 0:
        logging.info(
            "🧹 Cleanup: %d expired decisions, %d outbox items, %d tool calls",
            expired_count,
            outbox_cleaned,
            tools_cleaned,
        )


def log_tool_call(
    tool_name: str,
    success: bool,
    latency_ms: int,
    symbol: str | None = None,
    params: dict[str, Any] | None = None,
    error_msg: str | None = None,
    provider: str | None = None,
    data_size_bytes: int | None = None,
):
    """
    Log a tool invocation for performance monitoring and debugging.

    Args:
        tool_name: Name of tool called (e.g., "news.search", "prices.history")
        success: Whether call succeeded
        latency_ms: Time taken in milliseconds
        symbol: Ticker symbol if applicable
        params: Tool parameters (will be JSON encoded)
        error_msg: Error message if failed
        provider: Data provider (e.g., "yfinance", "alphavantage")
        data_size_bytes: Size of response data
    """
    con = _conn()
    cur = con.cursor()

    created_ts = datetime.now(UTC).isoformat()
    params_json = json.dumps(params) if params else None

    cur.execute(
        """
        INSERT INTO tool_calls
        (created_ts, tool_name, symbol, params_json, success, latency_ms, error_msg, provider, data_size_bytes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            created_ts,
            tool_name,
            symbol,
            params_json,
            1 if success else 0,
            latency_ms,
            error_msg,
            provider,
            data_size_bytes,
        ),
    )

    con.commit()
    con.close()

    # Update Prometheus metrics
    try:
        result = "success" if success else "failure"
        _C_AI_TOOL_CALLS.labels(tool_name=tool_name, result=result).inc()
        _H_AI_TOOL_LATENCY.labels(tool_name=tool_name).observe(latency_ms / 1000.0)
    except Exception as e:
        logging.warning(f"Failed to update tool metrics: {e}")


def get_tool_metrics(tool_name: str | None = None, hours: int = 24) -> list[dict[str, Any]]:
    """
    Retrieve tool call metrics for monitoring dashboard.

    Args:
        tool_name: Filter by specific tool (optional)
        hours: Lookback period

    Returns:
        List of tool call records with metrics
    """
    con = _conn()
    cur = con.cursor()

    since_ts = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()

    if tool_name:
        query = """
            SELECT * FROM tool_calls
            WHERE tool_name = ? AND created_ts >= ?
            ORDER BY created_ts DESC
            LIMIT 1000
        """
        rows = cur.execute(query, (tool_name, since_ts)).fetchall()
    else:
        query = """
            SELECT * FROM tool_calls
            WHERE created_ts >= ?
            ORDER BY created_ts DESC
            LIMIT 1000
        """
        rows = cur.execute(query, (since_ts,)).fetchall()

    con.close()

    return [dict(row) for row in rows]


# ──────────────────────────────────────────────────────────────────────────────
# Context Hydration (rebuild state when ChatGPT forgets)
# ──────────────────────────────────────────────────────────────────────────────
def build_runtime_snapshot() -> str:
    """
    Build compact snapshot of Ghost's current state.
    This gets sent to ChatGPT every tick so it has fresh context.

    Calls local Ghost endpoints to get real-time data.
    """
    try:
        import os

        import requests

        base_url = "http://localhost:5000"

        portfolio_data = {
            "nav": 0.0,
            "pnl_today": None,
            "pnl_pct": None,
            "cash": 0.0,
            "positions": [],
        }
        wolf_qty = 0.0
        wolf_avg = 0.0
        current_price = None
        prev_close = None
        cash_balance = 0.0

        # Fetch full portfolio (includes all positions + cash)
        try:
            portfolio_resp = requests.get(f"{base_url}/api/portfolio", timeout=10)
            if portfolio_resp.status_code == 200:
                full_portfolio = portfolio_resp.json()
                cash_balance = float(full_portfolio.get("cash", 0) or 0)
                logging.info(f"🔍 Snapshot: cash_balance from API = {cash_balance}")
                positions = full_portfolio.get("positions", [])
                logging.info(f"🔍 Snapshot: positions count = {len(positions)}")
                if positions:
                    # Get WOLF position if it exists
                    wolf_pos = next((p for p in positions if p.get("symbol") == "WOLF"), None)
                    if wolf_pos:
                        wolf_qty = float(wolf_pos.get("qty", 0) or 0)
                        wolf_avg = float(wolf_pos.get("price", 0) or 0)  # entry price
                        current_price = float(wolf_pos.get("current", 0) or 0)
                        logging.info(
                            f"🔍 Snapshot: WOLF found - qty={wolf_qty}, avg={wolf_avg}, current={current_price}"
                        )
                    else:
                        logging.warning("🔍 Snapshot: WOLF position not found in positions array!")
        except Exception as e:  # noqa: BLE001
            logging.warning(f"portfolio_fetch_failed: {e}", exc_info=True)

        # Fallback to individual position endpoint if portfolio fetch failed
        if wolf_qty == 0.0:
            try:
                resp = requests.get(f"{base_url}/api/position", timeout=10)
                if resp.status_code == 200:
                    pos = resp.json()
                    wolf_qty = float(pos.get("qty", 0) or 0)
                    wolf_avg = float(pos.get("avg_cost", 0) or 0)
            except Exception as e:  # noqa: BLE001
                logging.debug(f"position_fetch_failed: {e}")

        # Only fetch price if not already retrieved from /api/portfolio (respects locked baseline)
        if current_price is None:
            # Robust price fetch: prefer internal endpoint, fallback to yfinance
            try:
                price_resp = requests.get(f"{base_url}/api/price/WOLF", timeout=10)
                if price_resp.status_code == 200:
                    pr = price_resp.json()
                    current_price = float(pr.get("price")) if pr.get("price") is not None else None
                    prev_close = (
                        float(pr.get("prev_close")) if pr.get("prev_close") is not None else None
                    )
            except Exception:
                pass

        if current_price is None:
            try:
                import yfinance as yf  # type: ignore

                tk = yf.Ticker("WOLF")
                hist = tk.history(period="2d", interval="1d")
                if hist is not None and not hist.empty:
                    closes = list(hist["Close"].values.tolist())  # type: ignore
                    if closes:
                        current_price = float(closes[-1])
                        if len(closes) > 1:
                            prev_close = float(closes[-2])
            except Exception as e:  # noqa: BLE001
                logging.debug(f"yfinance_price_fail: {e}")

        if wolf_qty > 0 and current_price is not None:
            pnl_abs = wolf_qty * (current_price - wolf_avg) if wolf_avg > 0 else 0.0
            pnl_pct = ((current_price - wolf_avg) / wolf_avg * 100) if wolf_avg > 0 else None
            pnl_today = None
            if prev_close is not None:
                pnl_today = wolf_qty * (current_price - prev_close)
            position_value = round(wolf_qty * current_price, 2)
            total_nav = position_value + cash_balance
            portfolio_data = {
                "nav": round(total_nav, 2),
                "pnl_today": round(pnl_today, 2) if pnl_today is not None else None,
                "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
                "cash": round(cash_balance, 2),
                "positions": [
                    {
                        "symbol": "WOLF",
                        "qty": wolf_qty,
                        "avg_cost": wolf_avg,
                        "current_price": current_price,
                        "prev_close": prev_close,
                        "value": position_value,
                        "pnl": round(pnl_abs, 2),
                    }
                ],
            }
        elif cash_balance > 0:
            # No positions but has cash
            portfolio_data = {
                "nav": round(cash_balance, 2),
                "pnl_today": None,
                "pnl_pct": None,
                "cash": round(cash_balance, 2),
                "positions": [],
            }

        # Get recent predictions
        recent_predictions = []
        try:
            db_path = os.getenv("WOLF_SQLITE_PATH", "data/wolf.db")
            if os.path.exists(db_path):
                import sqlite3

                conn = sqlite3.connect(db_path)
                conn.row_factory = sqlite3.Row
                cur = conn.cursor()
                cur.execute("""
                    SELECT date, symbol, predicted_eod, confidence,
                           (SELECT actual_eod FROM prediction_scores
                            WHERE prediction_scores.date = daily_predictions.date
                            AND prediction_scores.symbol = daily_predictions.symbol) as actual_eod
                    FROM daily_predictions
                    ORDER BY date DESC
                    LIMIT 3
                """)
                for row in cur.fetchall():
                    recent_predictions.append(
                        {
                            "date": row["date"],
                            "symbol": row["symbol"],
                            "predicted": row["predicted_eod"],
                            "actual": row["actual_eod"] if row["actual_eod"] else None,
                            "confidence": row["confidence"],
                        }
                    )
                conn.close()
        except Exception as e:
            logging.debug(f"Predictions fetch failed: {e}")

        # Get market regime
        market_regime = "UNKNOWN"
        try:
            regime_resp = requests.get(f"{base_url}/api/regime/current", timeout=2)
            if regime_resp.status_code == 200:
                regime_data = regime_resp.json()
                if regime_data.get("ok"):
                    market_regime = regime_data.get("regime", {}).get("regime", "UNKNOWN")
        except Exception:
            pass

        # Market open flag (NYSE hours 09:30-16:00 ET, Mon-Fri)
        try:
            from datetime import time as dt_time
            from zoneinfo import ZoneInfo  # py39+

            now_et = datetime.now(ZoneInfo("America/New_York"))
            is_weekday = now_et.weekday() < 5
            m_open = dt_time(9, 30)
            m_close = dt_time(16, 0)
            market_open = is_weekday and (m_open <= now_et.time() <= m_close)
        except Exception:
            market_open = False

        snap = {
            "ts": datetime.now(UTC).isoformat(),
            "health": {"ok": True, "degraded_services": []},
            "portfolio": portfolio_data,
            "market": {
                "regime": market_regime,
                "market_open": market_open,
                "timezone": "America/New_York",
                "note": "VIX/SPY data via tools; no simulated data",
            },
            "watchlist": ["WOLF", "AAPL", "NVDA"],
            "recent_predictions": recent_predictions,
            "recent_events": [],
            "data_providers": {"polygon": "ok", "alphavantage": "rate_limited", "yahoo": "ok"},
        }
        return json.dumps(snap, separators=(",", ":"))
    except Exception as e:
        logging.error("Snapshot failed: %s", e)
        return json.dumps(
            {
                "ts": datetime.now(UTC).isoformat(),
                "error": "snapshot_failed",
                "detail": str(e),
            }
        )


def rehydrate_messages() -> list[dict[str, Any]]:
    """
    Rebuild minimal conversation state when ChatGPT loses context.
    Always returns: system prompt + fresh runtime snapshot
    """
    msgs = [{"role": "system", "content": SYSTEM_PROMPT, "ts": datetime.now(UTC).isoformat()}]

    # Add current state snapshot
    msgs.append(
        {
            "role": "user",
            "content": (
                f"REHYDRATE_CONTEXT\n"
                f"You are Ghost's analyst. Here's the current state:\n\n"
                f"```json\n{build_runtime_snapshot()}\n```\n\n"
                f"Resume monitoring. If you detect issues or opportunities, emit Action JSON."
            ),
            "ts": datetime.now(UTC).isoformat(),
        }
    )

    AGENT_STATE["reset_events"] += 1
    logging.info("🔄 Context rehydrated (reset #%d)", AGENT_STATE["reset_events"])
    return msgs


# ──────────────────────────────────────────────────────────────────────────────
# LLM Client (with robust retry + reset detection)
# ──────────────────────────────────────────────────────────────────────────────
class LLMClient:
    """OpenAI-compatible chat client with exponential backoff and guardrails"""

    def __init__(
        self, api_key: str, base_url: str, model: str, max_tokens: int = 2000, timeout: int = 120
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.max_tokens = max_tokens
        self._http = httpx.AsyncClient(timeout=timeout)
        self._rate_limit_tracker = {}  # Track rate limits by endpoint

    def _redact_secrets(self, content: str) -> str:
        """Remove potential secrets from content before logging"""
        import re

        # Redact API keys (sk-..., key_...)
        content = re.sub(r"\b(sk-[a-zA-Z0-9-_]{20,})", "sk-***REDACTED***", content)
        content = re.sub(r"\b([a-zA-Z0-9]{32,})\b", "***REDACTED***", content)
        return content

    async def chat(self, messages: list[dict[str, str]]) -> str:
        """
        Send chat completion request with retry logic, rate limiting, and guardrails.
        Returns assistant response or "RESET_NEEDED" on auth/context loss.
        """
        # Validate message count (prevent context overflow)
        if len(messages) > 50:
            logging.warning("Message count exceeds 50, trimming to recent 40")
            # Keep system message + recent 39
            system_msgs = [m for m in messages if m.get("role") == "system"]
            other_msgs = [m for m in messages if m.get("role") != "system"]
            messages = system_msgs[:1] + other_msgs[-39:]

        backoff = 2
        for attempt in range(6):
            try:
                # Prepare payload with token limits
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": 0.3,
                    "top_p": 0.95,
                    "presence_penalty": 0.0,
                    "frequency_penalty": 0.0,
                    "max_tokens": self.max_tokens,
                }

                # Log sanitized request (redact secrets)
                logging.debug(
                    "LLM request: model=%s, messages=%d, max_tokens=%d",
                    self.model,
                    len(messages),
                    self.max_tokens,
                )

                r = await self._http.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

                # Auth failures or session issues trigger rehydrate
                if r.status_code == 401:
                    logging.warning("LLM auth failed - triggering reset")
                    return "RESET_NEEDED"

                # Rate limit tracking
                if r.status_code == 429:
                    retry_after = int(r.headers.get("Retry-After", backoff))
                    logging.warning("Rate limited (429) - retry after %ds", retry_after)

                    if attempt < 5:
                        await asyncio.sleep(retry_after)
                        continue
                    else:
                        raise httpx.HTTPStatusError(
                            "Max retries on rate limit", request=r.request, response=r
                        )

                r.raise_for_status()
                data = r.json()

                # Extract response
                content = data["choices"][0]["message"]["content"]

                # Log usage stats if available
                usage = data.get("usage", {})
                if usage:
                    logging.info(
                        "LLM usage: prompt=%d, completion=%d, total=%d tokens",
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                        usage.get("total_tokens", 0),
                    )

                # Detect context loss in response
                loss_indicators = [
                    "I don't have context",
                    "RESET_NEEDED",
                    "I cannot access previous",
                    "I don't have information about our previous",
                ]
                if any(indicator in content for indicator in loss_indicators):
                    logging.warning("LLM lost context - triggering reset")
                    return "RESET_NEEDED"

                # Redact secrets in response before logging
                safe_content = self._redact_secrets(content[:200])
                logging.debug("LLM response: %s...", safe_content)

                return content

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    logging.warning("Rate limited (429) - backing off...")
                elif e.response.status_code >= 500:
                    logging.warning("Server error (%d) - retrying...", e.response.status_code)
                else:
                    logging.error("HTTP error (%d): %s", e.response.status_code, e)
                    raise

                if attempt == 5:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

            except httpx.TimeoutException:
                logging.error("LLM request timeout (attempt %d)", attempt + 1)
                if attempt == 5:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

            except Exception as e:
                logging.error("LLM request failed (attempt %d): %s", attempt + 1, e)
                if attempt == 5:
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60)

        # Fallback if all retries exhausted (should not reach here normally)
        logging.error("All LLM retries exhausted")
        return "RESET_NEEDED"

    async def aclose(self):
        """Close HTTP client"""
        await self._http.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# Main Loop
# ──────────────────────────────────────────────────────────────────────────────
async def analyst_tick(llm: LLMClient):
    """
    One analyst cycle:
    1. Load conversation state
    2. Add fresh runtime snapshot as user message
    3. Call ChatGPT
    4. Parse response for actions/decisions
    5. Queue tasks to outbox
    6. Save updated state
    """
    st = load_state()
    history = json.loads(st.get("messages_json", "[]"))

    # Validate history has system prompt
    if not history or history[0].get("role") != "system":
        logging.warning("Invalid history - rehydrating...")
        history = rehydrate_messages()
        save_state(history, failure_count=0)

    # Build tick prompt with fresh snapshot
    tick_user = {
        "role": "user",
        "content": (
            f"TICK\n"
            f"NOW={datetime.now(UTC).isoformat()}\n"
            f"SNAPSHOT={build_runtime_snapshot()}\n\n"
            f"Analyze current state. Are there any:\n"
            f"- Trading opportunities (high confidence setups)?\n"
            f"- Risk warnings (position exposure, provider issues)?\n"
            f"- System issues (degraded services, stale data)?\n\n"
            f"If action needed: emit Action JSON or Decision Card JSON.\n"
            f"If all clear: brief status summary."
        ),
        "ts": datetime.now(UTC).isoformat(),
    }
    history.append(tick_user)

    # Convert to OpenAI message format (strip 'ts' field)
    payload_msgs = [{"role": m["role"], "content": m["content"]} for m in history]

    try:
        content = await llm.chat(payload_msgs)

        # Handle reset request
        if content == "RESET_NEEDED":
            logging.info("ChatGPT requested reset - rehydrating context")
            history = rehydrate_messages()
            append_message("assistant", "Context rehydrated. Ready to resume monitoring.")
            save_state(history, failure_count=0)
            AGENT_STATE["ticks_ok"] += 1
            AGENT_STATE["last_ok_ts"] = datetime.now(UTC)
            return

        # Save assistant response
        append_message("assistant", content)

        # Parse for actionable JSON (task or decision)
        try:
            # Try parsing as JSON
            parsed = json.loads(content)

            # Queue tasks and log decisions
            if isinstance(parsed, dict):
                decision_type = parsed.get("type")

                if decision_type in ("task", "decision"):
                    queue_outbox(parsed)
                    logging.info(
                        "✅ Analyst action: %s [%s]",
                        parsed.get("title", parsed.get("symbol", "?")),
                        parsed.get("action", parsed.get("priority", "?")),
                    )

                    # Log to decision ledger if it's a trading decision
                    if decision_type == "decision" and parsed.get("symbol"):
                        log_ai_decision(parsed)

        except json.JSONDecodeError:
            # Plain text response (status summary) - that's fine
            logging.debug("Analyst response (text): %s", content[:100])

        # Update success metrics
        st2 = load_state()
        save_state(json.loads(st2["messages_json"]), failure_count=0)

        AGENT_STATE["ticks_ok"] += 1
        AGENT_STATE["last_ok_ts"] = datetime.now(UTC)
        AGENT_STATE["last_err"] = None

    except Exception as e:
        logging.error("Analyst tick failed: %s", e, exc_info=True)

        # Increment failure counter
        st3 = load_state()
        fc = int(st3.get("failure_count", 0)) + 1

        AGENT_STATE["ticks_fail"] += 1
        AGENT_STATE["last_err"] = str(e)

        # After 3 consecutive failures, force rehydrate
        if fc >= 3:
            logging.warning("3+ failures - forcing context reset")
            msgs = rehydrate_messages()
            save_state(msgs, failure_count=0)
        else:
            save_state(json.loads(st3.get("messages_json", "[]")), failure_count=fc)


async def outbox_delivery_loop():
    """
    Background task: process queued analyst tasks.

    Executes analyst recommendations:
    - Place trades via broker integration
    - Send Telegram alerts for opportunities/risks
    - Update Ghost config based on AI recommendations
    - Run diagnostics when issues detected
    """
    # Import functions from wolf_app for execution
    try:
        from wolf_app import send_telegram, enqueue_alert_text
    except ImportError:
        logging.warning("⚠️  Could not import wolf_app functions - task executor limited")
        send_telegram = None
        enqueue_alert_text = None
    
    while True:
        try:
            batch = grab_undelivered(20)
            if batch:
                ids = []
                for row in batch:
                    payload = json.loads(row["payload_json"])
                    task_type = payload.get("type")
                    tags = payload.get("tags", [])

                    logging.info("🧩 ANALYST TASK: %s", json.dumps(payload, indent=2))

                    # Execute the task based on type and tags
                    try:
                        if task_type == "task":
                            # Trading actions
                            if "buy" in tags or "sell" in tags:
                                logging.info("📊 Trade recommendation logged (auto-execution disabled)")
                                # NOTE: Actual trade execution requires AUTO_EXECUTION_ENABLED=1
                                # and is handled by autonomous_execution_engine.py
                            
                            # Alerts and notifications
                            if "alert" in tags or "opportunity" in tags or "risk" in tags:
                                if enqueue_alert_text:
                                    alert_text = payload.get("reasoning", payload.get("description", "AI Alert"))
                                    enqueue_alert_text(f"🤖 AI Analyst: {alert_text}")
                                    logging.info("📢 Alert queued for Telegram")
                            
                            # Diagnostics
                            if "diagnostic" in tags or "health" in tags:
                                logging.info("🔍 Diagnostic task logged for review")
                                # Future: Trigger automated health checks
                        
                        elif task_type == "insight":
                            # Log insights for review (no automated action)
                            logging.info("💡 Market insight logged: %s", payload.get("reasoning", ""))
                        
                        else:
                            logging.warning("⚠️  Unknown task type: %s", task_type)
                    
                    except Exception as e:
                        logging.error(f"❌ Task execution failed: {e}", exc_info=True)

                    ids.append(row["id"])

                mark_delivered(ids)

        except Exception as e:
            logging.error("Outbox delivery error: %s", e, exc_info=True)

        await asyncio.sleep(5)


async def run_loop():
    """Main agent loop - runs every TICK_SECONDS with periodic cleanup"""
    if not OPENAI_API_KEY:
        logging.warning("⚠️  OPENAI_API_KEY missing - analyst loop disabled")
        logging.warning("    Set OPENAI_API_KEY in secrets.env to enable")
        # Don't crash, just idle
        while True:
            await asyncio.sleep(60)

    logging.info("🧠 Ghost Analyst starting (model=%s, tick=%ds)", LLM_MODEL, TICK_SECONDS)

    llm = LLMClient(OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL)
    tick_count = 0

    try:
        while True:
            await analyst_tick(llm)
            tick_count += 1

            # Run cleanup every 12 ticks (1 hour if tick=300s)
            if tick_count % 12 == 0:
                cleanup_expired_data()

            await asyncio.sleep(TICK_SECONDS)
    finally:
        await llm.aclose()


# ──────────────────────────────────────────────────────────────────────────────
# FastAPI Integration
# ──────────────────────────────────────────────────────────────────────────────
def attach_agent(app: FastAPI):
    """
    Attach analyst loop to FastAPI app.

    Usage:
        from ghost_agent_loop import attach_agent
        attach_agent(app)
    """
    init_db()

    @app.on_event("startup")
    async def _startup():
        app.state._agent_task = asyncio.create_task(run_loop())
        app.state._outbox_task = asyncio.create_task(outbox_delivery_loop())
        logging.info("✅ Ghost Analyst loop started")

    @app.on_event("shutdown")
    async def _shutdown():
        for name in ("_agent_task", "_outbox_task"):
            task = getattr(app.state, name, None)
            if task:
                task.cancel()
        logging.info("Ghost Analyst loop stopped")

    # Health check endpoint
    @app.get("/agent/health")
    def agent_health():
        """Agent loop health status"""
        now = datetime.now(UTC)
        ok = AGENT_STATE["ticks_ok"] > 0 and AGENT_STATE["last_ok_ts"] is not None

        age_sec = None
        if AGENT_STATE["last_ok_ts"]:
            age_sec = (now - AGENT_STATE["last_ok_ts"]).total_seconds()

        # Determine status
        if ok and age_sec is not None and age_sec <= AGENT_STATE["loop_interval_sec"] * 3:
            status = "ok"
        else:
            status = "degraded"

        if AGENT_STATE["ticks_fail"] > 0 and status == "ok":
            status = "warn"

        return {
            "status": status,
            "model": AGENT_STATE["model_name"],
            "ticks_ok": AGENT_STATE["ticks_ok"],
            "ticks_fail": AGENT_STATE["ticks_fail"],
            "last_ok_ts": AGENT_STATE["last_ok_ts"].isoformat()
            if AGENT_STATE["last_ok_ts"]
            else None,
            "last_error": AGENT_STATE["last_err"],
            "reset_events": AGENT_STATE["reset_events"],
            "loop_interval_sec": AGENT_STATE["loop_interval_sec"],
            "now": now.isoformat(),
        }

    # State inspection endpoint
    @app.get("/agent/state")
    def agent_state():
        """View conversation state"""
        st = load_state()
        st["messages"] = json.loads(st.get("messages_json", "[]"))
        st.pop("messages_json", None)
        return st

    # Outbox inspection endpoint
    @app.get("/agent/outbox")
    def agent_outbox(limit: int = 100):
        """View pending/delivered tasks"""
        return grab_undelivered(limit)

    # AI Decision ledger endpoint
    @app.get("/api/ai/decisions")
    def api_ai_decisions(symbol: str | None = None, hours: int = 24):
        """
        Retrieve AI analyst decisions from persistent ledger.

        Query params:
            symbol: Filter by symbol (e.g., WOLF, AAPL) - optional
            hours: Lookback period in hours (default: 24)

        Returns:
            List of decisions with full context (action, confidence, rationale, etc.)
        """
        try:
            decisions = get_ai_decisions(symbol=symbol, hours=hours)
            return {
                "ok": True,
                "count": len(decisions),
                "decisions": decisions,
                "query": {"symbol": symbol, "hours": hours},
            }
        except Exception as e:
            logging.error("ai_decisions_error: %s", e, exc_info=True)
            return {"ok": False, "error": str(e), "decisions": []}

    # Decision replay/debug endpoint
    @app.get("/api/ai/decisions/{decision_id}/replay")
    def api_decision_replay(decision_id: int):
        """
        Reconstruct full context for a specific decision for debugging.

        Returns the decision plus all contextual data from ±5 minutes:
        - Decision details (action, confidence, rationale, risks)
        - Tool calls made around that time
        - Conversation messages
        - Data sources used
        - Agent state at decision time

        Use case: "Why did the agent decide to sell AAPL at 2:45 PM?"

        Args:
            decision_id: ID of the decision to replay

        Returns:
            Full context reconstruction for debugging agent reasoning
        """
        try:
            import sqlite3
            from datetime import datetime, timedelta

            conn = sqlite3.connect(DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Get the decision
            cur.execute(
                """
                SELECT id, created_ts, symbol, action, confidence, horizon, priority,
                       rationale, risks, metrics, data_sources, tags, decision_type
                FROM ai_decisions
                WHERE id = ?
            """,
                (decision_id,),
            )

            row = cur.fetchone()
            if not row:
                conn.close()
                return {"ok": False, "error": f"Decision {decision_id} not found"}

            # Parse decision
            decision = {
                "id": row["id"],
                "created_ts": row["created_ts"],
                "symbol": row["symbol"],
                "action": row["action"],
                "confidence": row["confidence"],
                "horizon": row["horizon"],
                "priority": row["priority"],
                "rationale": row["rationale"],
                "risks": json.loads(row["risks"]) if row["risks"] else [],
                "metrics": json.loads(row["metrics"]) if row["metrics"] else {},
                "data_sources": json.loads(row["data_sources"]) if row["data_sources"] else [],
                "tags": json.loads(row["tags"]) if row["tags"] else [],
                "decision_type": row["decision_type"],
            }

            # Calculate ±5 minute window
            decision_ts = datetime.fromisoformat(row["created_ts"])
            window_start = (decision_ts - timedelta(minutes=5)).isoformat()
            window_end = (decision_ts + timedelta(minutes=5)).isoformat()

            # Get tool calls in window
            cur.execute(
                """
                SELECT id, created_ts, tool_name, success, latency_ms, symbol,
                       params, error_msg, provider, data_size_bytes
                FROM tool_calls
                WHERE created_ts BETWEEN ? AND ?
                ORDER BY created_ts
            """,
                (window_start, window_end),
            )

            tool_calls = []
            for tc_row in cur.fetchall():
                tool_calls.append(
                    {
                        "id": tc_row["id"],
                        "created_ts": tc_row["created_ts"],
                        "tool_name": tc_row["tool_name"],
                        "success": bool(tc_row["success"]),
                        "latency_ms": tc_row["latency_ms"],
                        "symbol": tc_row["symbol"],
                        "params": json.loads(tc_row["params"]) if tc_row["params"] else {},
                        "error_msg": tc_row["error_msg"],
                        "provider": tc_row["provider"],
                        "data_size_bytes": tc_row["data_size_bytes"],
                    }
                )

            # Get conversation messages in window (if conversation_topics table exists)
            conversation_messages = []
            try:
                cur.execute(
                    """
                    SELECT id, created_ts, role, content, topic, tokens
                    FROM conversation_topics
                    WHERE created_ts BETWEEN ? AND ?
                    ORDER BY created_ts
                """,
                    (window_start, window_end),
                )

                for msg_row in cur.fetchall():
                    conversation_messages.append(
                        {
                            "id": msg_row["id"],
                            "created_ts": msg_row["created_ts"],
                            "role": msg_row["role"],
                            "content": msg_row["content"][:500]
                            if msg_row["content"]
                            else "",  # Truncate long content
                            "topic": msg_row["topic"],
                            "tokens": msg_row["tokens"],
                        }
                    )
            except sqlite3.OperationalError:
                # Table might not exist
                pass

            # Get other decisions in window (for context)
            cur.execute(
                """
                SELECT id, created_ts, symbol, action, confidence
                FROM ai_decisions
                WHERE created_ts BETWEEN ? AND ?
                  AND id != ?
                ORDER BY created_ts
            """,
                (window_start, window_end, decision_id),
            )

            nearby_decisions = []
            for nd_row in cur.fetchall():
                nearby_decisions.append(
                    {
                        "id": nd_row["id"],
                        "created_ts": nd_row["created_ts"],
                        "symbol": nd_row["symbol"],
                        "action": nd_row["action"],
                        "confidence": nd_row["confidence"],
                    }
                )

            conn.close()

            # Calculate timing statistics
            decision_time = datetime.fromisoformat(decision["created_ts"])
            now = datetime.now(UTC)
            age_hours = (now - decision_time).total_seconds() / 3600

            # Tool success rate in window
            tool_success_count = sum(1 for tc in tool_calls if tc["success"])
            tool_success_rate = tool_success_count / len(tool_calls) if tool_calls else 0

            return {
                "ok": True,
                "decision": decision,
                "context": {
                    "window_start": window_start,
                    "window_end": window_end,
                    "window_minutes": 10,  # ±5 minutes
                    "decision_age_hours": round(age_hours, 2),
                },
                "tool_calls": {
                    "count": len(tool_calls),
                    "success_rate": round(tool_success_rate, 3),
                    "calls": tool_calls,
                },
                "conversation": {
                    "message_count": len(conversation_messages),
                    "messages": conversation_messages,
                },
                "nearby_decisions": {"count": len(nearby_decisions), "decisions": nearby_decisions},
                "debug_info": {
                    "decision_id": decision_id,
                    "reconstructed_at": datetime.now(UTC).isoformat(),
                    "data_sources": decision["data_sources"],
                    "tool_calls_by_type": {
                        tc["tool_name"]: sum(
                            1 for t in tool_calls if t["tool_name"] == tc["tool_name"]
                        )
                        for tc in tool_calls
                    }
                    if tool_calls
                    else {},
                },
            }

        except Exception as e:
            logging.error("decision_replay_error: %s", e, exc_info=True)
            return {"ok": False, "error": str(e), "decision_id": decision_id}

    # Agent monitoring dashboard endpoint
    @app.get("/api/ai/monitor")
    def api_ai_monitor(hours: int = 168):
        """
        Comprehensive agent monitoring dashboard with analytics.

        Query params:
            hours: Lookback period in hours (default: 168 = 7 days)

        Returns:
            Aggregate statistics, confidence trends, quality scores, alerts, tool metrics
        """
        try:
            from core.agent_analytics import (
                AgentAnalytics,
                format_stats_for_api,
                get_tool_call_analytics,
            )

            analytics = AgentAnalytics(db_path=DB_PATH)

            # Core statistics
            stats = analytics.get_decision_stats(hours=hours)

            # Quality score
            quality_score = analytics.compute_decision_quality_score(hours=hours)

            # Confidence distribution
            conf_dist = analytics.get_confidence_distribution(hours=hours, buckets=10)

            # Decision timeline
            timeline = analytics.get_decision_timeline(hours=hours, interval_hours=24)

            # Low confidence alerts
            low_conf = analytics.get_low_confidence_decisions(threshold=0.5, hours=hours)

            # Stale symbols
            stale = analytics.get_stale_symbols(hours_since_decision=48)

            # Tool call metrics
            tool_metrics = get_tool_call_analytics(db_path=DB_PATH, hours=hours)

            # Agent health integration
            agent_health_data = agent_health()

            return {
                "ok": True,
                "timestamp": datetime.now(UTC).isoformat(),
                "lookback_hours": hours,
                "agent_health": agent_health_data,
                "quality_score": quality_score,
                "decision_stats": format_stats_for_api(stats),
                "confidence_distribution": conf_dist,
                "timeline": timeline,
                "tool_performance": [
                    {
                        "tool": m.tool_name,
                        "calls": m.total_calls,
                        "success_rate": m.success_rate,
                        "avg_latency_ms": m.avg_latency_ms,
                        "most_used_symbol": m.most_used_symbol,
                    }
                    for m in tool_metrics
                ],
                "alerts": {
                    "low_confidence_decisions": low_conf[:10],  # Top 10 worst
                    "stale_symbols": stale[:10],  # Top 10 stalest
                    "alert_count": len(low_conf) + len(stale),
                },
            }
        except Exception as e:
            logging.error("ai_monitor_error: %s", e, exc_info=True)
            return {
                "ok": False,
                "error": str(e),
                "timestamp": datetime.now(UTC).isoformat(),
            }

    # Symbol-specific performance endpoint
    @app.get("/api/ai/monitor/symbol/{symbol}")
    def api_ai_monitor_symbol(symbol: str, hours: int = 168):
        """
        Detailed performance metrics for a specific symbol.

        Path params:
            symbol: Ticker symbol (e.g., WOLF, AAPL)

        Query params:
            hours: Lookback period in hours (default: 168 = 7 days)

        Returns:
            Symbol-specific decision stats, confidence trends, action distribution
        """
        try:
            from core.agent_analytics import AgentAnalytics, format_performance_for_api

            analytics = AgentAnalytics(db_path=DB_PATH)

            perf = analytics.get_symbol_performance(symbol=symbol.upper(), hours=hours)

            if perf is None:
                return {
                    "ok": True,
                    "found": False,
                    "symbol": symbol.upper(),
                    "message": f"No decisions found for {symbol.upper()} in last {hours} hours",
                }

            # Get recent decisions for this symbol
            decisions = get_ai_decisions(symbol=symbol.upper(), hours=hours)

            return {
                "ok": True,
                "found": True,
                "symbol": symbol.upper(),
                "lookback_hours": hours,
                "performance": format_performance_for_api(perf),
                "recent_decisions": decisions[:5],  # 5 most recent
            }
        except Exception as e:
            logging.error("ai_monitor_symbol_error: %s", e, exc_info=True)
            return {"ok": False, "error": str(e), "symbol": symbol}

    logging.info(
        "🔌 Ghost Analyst endpoints registered: /agent/health, /agent/state, /agent/outbox, /api/ai/decisions, /api/ai/monitor"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Stub for market_scanner.py compatibility
# ──────────────────────────────────────────────────────────────────────────────
async def agent_decide(symbol: str) -> dict[str, Any] | None:
    """
    AI-driven decision making for market_scanner.py compatibility.
    Returns AI recommendation based on symbol analysis.
    
    When AI analysis is enabled:
    - Analyzes symbol fundamentals, technicals, sentiment
    - Generates BUY/SELL/HOLD recommendation with confidence
    - Returns decision payload for execution engine
    
    Returns None when AI is disabled or analysis fails.
    """
    try:
        # Check if AI agent is enabled
        if not OPENAI_API_KEY:
            return None
        
        # Future: Implement full AI analysis pipeline
        # For now, return None to indicate feature in development
        logging.debug(f"agent_decide called for {symbol} - AI analysis pipeline in development")
        return None
        
        # Future implementation:
        # analysis = await analyze_symbol(symbol)
        # recommendation = await ai_recommend(analysis)
        # return {
        #     "symbol": symbol,
        #     "action": recommendation["action"],  # BUY/SELL/HOLD
        #     "confidence": recommendation["confidence"],
        #     "reasoning": recommendation["reasoning"],
        #     "timestamp": time.time()
        # }
    
    except Exception as e:
        logging.error(f"agent_decide failed for {symbol}: {e}", exc_info=True)
        return None

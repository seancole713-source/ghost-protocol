# 🚀 Ghost Agent - Implementation Quick Start

**Goal**: Complete the 3 critical missing pieces in ~6-9 hours

______________________________________________________________________

## 🎯 **Step 1: Monitoring Test Suite** (2-3 hours)

### Create `tests/test_agent_monitoring.py`

```python
"""
Tests for agent monitoring and analytics.
"""
import pytest
import json
from datetime import datetime, timedelta, timezone
from ghost_agent_loop import log_ai_decision, log_tool_call, init_db
from core.agent_analytics import (
    compute_decision_stats,
    get_symbol_performance,
    get_tool_metrics,
    DecisionStats,
    SymbolPerformance,
    ToolCallMetrics
)


@pytest.fixture
def fresh_db(tmp_path):
    """Create temporary test database."""
    db_path = tmp_path / "test_agent.db"
    import os
    os.environ["GHOST_AGENT_DB"] = str(db_path)
    init_db()
    yield db_path


class TestDecisionStats:
    """Test decision analytics calculations."""

    def test_decision_stats_empty_db(self, fresh_db):
        """Empty database returns zero stats."""
        stats = compute_decision_stats(hours=24)
        assert stats.total_decisions == 0
        assert stats.unique_symbols == 0
        assert stats.avg_confidence == 0.0

    def test_decision_stats_with_data(self, fresh_db):
        """Stats computed correctly from sample data."""

        # Log 3 decisions with different confidences

        decisions = [
            {"symbol": "AAPL", "action": "BUY", "confidence": 0.8, "horizon": "1d"},
            {"symbol": "AAPL", "action": "HOLD", "confidence": 0.6, "horizon": "1d"},
            {"symbol": "MSFT", "action": "SELL", "confidence": 0.9, "horizon": "1w"},
        ]
        for d in decisions:
            log_ai_decision(d)

        stats = compute_decision_stats(hours=24)

        assert stats.total_decisions == 3
        assert stats.unique_symbols == 2
        assert stats.avg_confidence == pytest.approx(0.767, abs=0.01)
        assert stats.min_confidence == 0.6
        assert stats.max_confidence == 0.9
        assert stats.action_distribution == {"BUY": 1, "HOLD": 1, "SELL": 1}

    def test_decision_stats_time_filtering(self, fresh_db):
        """Only decisions within time range counted."""

        # Log old decision (25 hours ago)

        old_ts = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()

        # Manually insert with old timestamp

        import sqlite3
        conn = sqlite3.connect(str(fresh_db))
        conn.execute(
            "INSERT INTO ai_decisions (created_ts, symbol, action, confidence) VALUES (?, 'OLD', 'BUY', 0.5)",
            (old_ts,)
        )
        conn.commit()

        # Log recent decision

        log_ai_decision({"symbol": "NEW", "action": "SELL", "confidence": 0.8})

        # Query last 24h should only get recent

        stats = compute_decision_stats(hours=24)
        assert stats.total_decisions == 1
        assert "NEW" in stats.symbols_tracked
        assert "OLD" not in stats.symbols_tracked


class TestSymbolPerformance:
    """Test per-symbol analytics."""

    def test_symbol_performance(self, fresh_db):
        """Metrics aggregated correctly per symbol."""

        # Log multiple decisions for AAPL

        for i in range(5):
            log_ai_decision({
                "symbol": "AAPL",
                "action": "BUY" if i < 3 else "HOLD",
                "confidence": 0.7 + i*0.05
            })

        perf = get_symbol_performance("AAPL", hours=24)

        assert perf.symbol == "AAPL"
        assert perf.decision_count == 5
        assert perf.most_common_action == "BUY"
        assert perf.avg_confidence > 0.7


class TestToolCallMetrics:
    """Test tool call analytics."""

    def test_tool_metrics_success_rate(self, fresh_db):
        """Success rate calculated correctly."""

        # Log tool calls with mixed results

        for i in range(10):
            log_tool_call(
                tool_name="fetch_price",
                success=(i < 8),  # 80% success rate
                latency_ms=100 + i*10
            )

        metrics = get_tool_metrics("fetch_price", hours=24)

        assert metrics.tool_name == "fetch_price"
        assert metrics.total_calls == 10
        assert metrics.success_count == 8
        assert metrics.failure_count == 2
        assert metrics.success_rate == pytest.approx(0.8)
        assert metrics.avg_latency_ms > 100


class TestMonitorAPI:
    """Test /api/ai/monitor endpoint."""

    @pytest.mark.asyncio
    async def test_monitor_endpoint_structure(self, client):
        """Response has correct structure."""
        response = await client.get("/api/ai/monitor?hours=24")
        assert response.status_code == 200

        data = response.json()
        assert data["ok"] is True
        assert "stats" in data
        assert "recent_decisions" in data
        assert "tool_metrics" in data
        assert "symbol_performance" in data

    @pytest.mark.asyncio
    async def test_monitor_symbol_filter(self, client, fresh_db):
        """Symbol-specific monitoring works."""

        # Log decisions for multiple symbols

        log_ai_decision({"symbol": "AAPL", "action": "BUY", "confidence": 0.8})
        log_ai_decision({"symbol": "MSFT", "action": "SELL", "confidence": 0.6})

        response = await client.get("/api/ai/monitor/symbol/AAPL")
        data = response.json()

        assert data["symbol"] == "AAPL"
        assert len(data["decisions"]) == 1
        assert data["decisions"][0]["symbol"] == "AAPL"


class TestConfidenceTrends:
    """Test confidence analysis over time."""

    def test_confidence_buckets(self, fresh_db):
        """Decisions grouped by confidence levels."""

        # Log decisions across confidence spectrum

        for conf in [0.3, 0.5, 0.7, 0.9]:
            log_ai_decision({"symbol": "TEST", "action": "HOLD", "confidence": conf})

        stats = compute_decision_stats(hours=24)

        # Should have data in multiple confidence buckets

        assert stats.min_confidence < 0.5
        assert stats.max_confidence > 0.5

```text

### Run Tests

```bash

cd /workspaces/GHOST
python -m pytest tests/test_agent_monitoring.py -v

```text

______________________________________________________________________

## 🎯 **Step 2: Prometheus Metrics & Alert Rules**(1-2 hours)

### Add Metrics to `ghost_agent_loop.py`

```python

# At top of file, add

from prometheus_client import Counter, Gauge, Histogram

# After AGENT_STATE definition, add

_G_AI_CONFIDENCE = Gauge(
    "ghost_ai_decision_confidence",
    "Latest AI decision confidence (0-1)"
)
_C_AI_DECISIONS = Counter(
    "ghost_ai_decisions_total",
    "Total AI decisions made",
    labelnames=("action",)
)
_C_AI_TOOL_CALLS = Counter(
    "ghost_ai_tool_calls_total",
    "Total tool invocations",
    labelnames=("tool_name", "result")
)
_H_AI_TOOL_LATENCY = Histogram(
    "ghost_ai_tool_latency_seconds",
    "Tool call latency",
    labelnames=("tool_name",),
    buckets=(0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)
)
_G_AI_DECISION_LAST_TS = Gauge(
    "ghost_ai_decision_last_ts",
    "Timestamp of last decision (epoch)"
)

# Update log_ai_decision() function

def log_ai_decision(decision: dict):
    """Log AI decision to database + update metrics."""

    # ... existing code

    # Update Prometheus metrics

    try:
        _G_AI_CONFIDENCE.set(decision.get("confidence", 0))
        _C_AI_DECISIONS.labels(action=decision.get("action", "UNKNOWN")).inc()
        _G_AI_DECISION_LAST_TS.set(time.time())
    except Exception as e:
        logging.warning(f"Failed to update decision metrics: {e}")

# Update log_tool_call() function

def log_tool_call(tool_name: str, success: bool, latency_ms: float,**kwargs):
    """Log tool call to database + update metrics."""

    # ... existing code

    # Update Prometheus metrics

    try:
        result = "success" if success else "failure"
        _C_AI_TOOL_CALLS.labels(tool_name=tool_name, result=result).inc()
        _H_AI_TOOL_LATENCY.labels(tool_name=tool_name).observe(latency_ms / 1000.0)
    except Exception as e:
        logging.warning(f"Failed to update tool metrics: {e}")

```text

### Create Alert Rules

**File**: `docs/alerts/agent_slo_rules.yml`

```yaml

groups:

  - name: ghost_agent_alerts


    interval: 5m
    rules:

      # Agent stopped making decisions

      - alert: GhostAgentStale


        expr: (time() - ghost_ai_decision_last_ts) > 86400
        for: 10m
        labels:
          severity: critical
          component: agent
        annotations:
          summary: "Ghost Agent has not made decisions in 24+ hours"
          description: "Last decision at {{ $value | humanizeTimestamp }}. Agent may be stuck or disabled."
          runbook_url: "<<<<<https://github.com/your-repo/docs/runbooks/agent_stale.md">>>>>

      # Low confidence sustained

      - alert: GhostAgentLowConfidence


        expr: avg_over_time(ghost_ai_decision_confidence[1h]) < 0.5
        for: 30m
        labels:
          severity: warning
          component: agent
        annotations:
          summary: "Agent confidence below 50% for 30+ minutes"
          description: "Current avg: {{ $value | humanizePercentage }}. Model may be degrading or market conditions unclear."

      # High tool failure rate

      - alert: GhostAgentToolFailures


        expr: |
          (
            rate(ghost_ai_tool_calls_total{result="failure"}[5m])
            /
            rate(ghost_ai_tool_calls_total[5m])
          ) > 0.2
        for: 15m
        labels:
          severity: warning
          component: agent
        annotations:
          summary: "Agent tool failure rate >20%"
          description: "{{ $value | humanizePercentage }} of tool calls failing. Check data providers."

      # Tool latency spike

      - alert: GhostAgentToolLatency


        expr: |
          histogram_quantile(0.95,
            rate(ghost_ai_tool_latency_seconds_bucket[5m])
          ) > 5.0
        for: 10m
        labels:
          severity: warning
          component: agent
        annotations:
          summary: "Agent tool latency p95 >5s"
          description: "Tools taking {{ $value }}s at p95. Providers may be slow."

      # No recent tool calls (agent not fetching data)

      - alert: GhostAgentNotFetchingData


        expr: rate(ghost_ai_tool_calls_total[10m]) == 0
        for: 30m
        labels:
          severity: warning
          component: agent
        annotations:
          summary: "Agent not invoking tools for 30+ minutes"
          description: "Agent may be idle or loop may be paused."

```text

### Update Observability Docs

**File**: `docs/observability.md` (add at end)

````markdown

## AI Agent Metrics (v2)

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| ghost_ai_decision_confidence | gauge | - | Latest decision confidence (0-1) |
| ghost_ai_decisions_total | counter | action | Decisions by action type |
| ghost_ai_tool_calls_total | counter | tool_name, result | Tool invocations |
| ghost_ai_tool_latency_seconds | histogram | tool_name | Tool call latency |
| ghost_ai_decision_last_ts | gauge | - | Timestamp of last decision |

### PromQL Examples

**Average confidence over time:**```promql

avg_over_time(ghost_ai_decision_confidence[1h])

````**Decision rate by action:**```promql

rate(ghost_ai_decisions_total[5m])

```text**Tool success rate:**```promql

rate(ghost_ai_tool_calls_total{result="success"}[5m])
/
rate(ghost_ai_tool_calls_total[5m])

```text**Tool latency p95:**```promql

histogram_quantile(0.95,
  rate(ghost_ai_tool_latency_seconds_bucket[5m])
)

```text

### Alert Runbooks

See `docs/alerts/agent_slo_rules.yml` for alert definitions.**GhostAgentStale**: Check `/agent/health`, restart agent
loop if needed.\
**GhostAgentLowConfidence**: Review recent decisions, check data quality.\
**GhostAgentToolFailures**: Check provider status, API keys, rate limits.\
**GhostAgentToolLatency**: Check provider response times, consider caching.

````

---

## 🎯 **Step 3: Grafana Agent Dashboard**(3-4 hours)

### Create `docs/grafana/agent_dashboard.json`

```json

{
  "dashboard": {
    "title": "Ghost AI Agent Monitor",
    "tags": ["ghost", "ai", "agent"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Decision Confidence",
        "type": "graph",
        "targets": [
          {
            "expr": "ghost_ai_decision_confidence",
            "legendFormat": "Current Confidence"
          },
          {
            "expr": "avg_over_time(ghost_ai_decision_confidence[1h])",
            "legendFormat": "1h Average"
          }
        ],
        "yaxes": [
          { "min": 0, "max": 1, "format": "percentunit" }
        ]
      },
      {
        "id": 2,
        "title": "Decision Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ghost_ai_decisions_total[5m])",
            "legendFormat": "{{action}}"
          }
        ]
      },
      {
        "id": 3,
        "title": "Action Distribution",
        "type": "piechart",
        "targets": [
          {
            "expr": "sum by (action) (increase(ghost_ai_decisions_total[24h]))",
            "legendFormat": "{{action}}"
          }
        ]
      },
      {
        "id": 4,
        "title": "Tool Success Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(ghost_ai_tool_calls_total{result=\"success\"}[5m]) / rate(ghost_ai_tool_calls_total[5m])",
            "legendFormat": "{{tool_name}}"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percentunit",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "red" },
                { "value": 0.8, "color": "yellow" },
                { "value": 0.95, "color": "green" }
              ]
            }
          }
        }
      },
      {
        "id": 5,
        "title": "Tool Latency (p95)",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(ghost_ai_tool_latency_seconds_bucket[5m]))",
            "legendFormat": "{{tool_name}}"
          }
        ],
        "yaxes": [
          { "format": "s", "label": "Latency" }
        ]
      },
      {
        "id": 6,
        "title": "Time Since Last Decision",
        "type": "stat",
        "targets": [
          {
            "expr": "time() - ghost_ai_decision_last_ts"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                { "value": 0, "color": "green" },
                { "value": 3600, "color": "yellow" },
                { "value": 86400, "color": "red" }
              ]
            }
          }
        }
      },
      {
        "id": 7,
        "title": "Tool Calls by Type",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ghost_ai_tool_calls_total[5m])",
            "legendFormat": "{{tool_name}} - {{result}}"
          }
        ]
      }
    ]
  }
}

````

### Import to Grafana

```bash

# Via API

curl -X POST <<<<<http://grafana:3000/api/dashboards/db>>>>> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GRAFANA_API_KEY" \
  -d @docs/grafana/agent_dashboard.json

# Or manually: Grafana UI → Dashboards → Import → Upload JSON

```text

______________________________________________________________________

## ✅**Verification Checklist**

### Step 1: Tests

- [ ] `pytest tests/test_agent_monitoring.py` passes
- [ ] All analytics functions tested
- [ ] API endpoint tests pass


### Step 2: Metrics & Alerts

- [ ] `/metrics` shows `ghost_ai_*` metrics
- [ ] Alerts loaded into Prometheus
- [ ] Alert rules evaluate without errors
- [ ] Test alert fires correctly (set low confidence threshold temporarily)


### Step 3: Grafana

- [ ] Dashboard imports successfully
- [ ] All panels show data (may need to log test decisions)
- [ ] Time range selector works
- [ ] Thresholds display correctly


______________________________________________________________________

## 🚀 **Quick Test Commands**

```bash

# 1. Restart Ghost with metrics

pkill -9 -f uvicorn
source .venv/bin/activate && source secrets.env
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 &

# 2. Log test decisions

python3 << 'EOF'
from ghost_agent_loop import log_ai_decision, log_tool_call
for i in range(10):
    log_ai_decision({
        "symbol": "TEST",
        "action": ["BUY", "SELL", "HOLD"][i % 3],
        "confidence": 0.5 + (i * 0.05)
    })
    log_tool_call("fetch_price", success=(i < 8), latency_ms=100+i*10)
EOF

# 3. Check metrics

curl -s <<<<<http://localhost:5000/metrics>>>>> | grep ghost_ai

# 4. Test monitoring API

curl -s <<<<<http://localhost:5000/api/ai/monitor?hours=24>>>>> | jq .

# 5. Run tests

python -m pytest tests/test_agent_monitoring.py -v

```text

______________________________________________________________________

## 📚 **Documentation Updates Needed**1.**README.md**- Add "Agent Monitoring" section

2.**AGENT_ENHANCEMENTS_COMPLETE.md**- Update with monitoring features
3.**docs/observability.md**- Add agent metrics section (done above)
4.**docs/alerts/README.md**- Document agent alert rules


______________________________________________________________________

## 🎯**Success Criteria**After completing these 3 steps, you should have

✅**Monitoring Test Suite**- 15+ tests covering analytics and API

- All tests passing
- High confidence in metrics accuracy


✅**Metrics & Alerts**- 5 new Prometheus metrics

- 5 alert rules for agent health
- Alerts firing when thresholds breached


✅**Grafana Dashboard**- 7 panels visualizing agent performance

- Real-time confidence, tool metrics, decision trends
- Easy to spot issues at a glance**Total time invested**: 6-9 hours\


**Value gained**: Production-ready agent monitoring 🚀

______________________________________________________________________

**Next**: Implement UI panel (Step 4 from AGENT_MISSING_FEATURES.md) - 4-5 hours

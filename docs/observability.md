# Observability v1

This document consolidates metrics, example PromQL, and alert recommendations introduced in Observability v1.

## Metrics Overview

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| ghost_cockpit_snapshot_build_seconds | histogram | le | Snapshot build latency (server side) |
| ghost_cockpit_snapshot_failures_total | counter | - | Count of snapshot build failures |
| ghost_provider_fetch_seconds | histogram | provider, le | Upstream provider fetch latency |
| ghost_price_cache_hits / misses | counter | kind | Local price cache performance |
| ghost_price_provider_fetch_total | counter | kind, provider, result | Fetch attempts outcome |
| ghost_up | gauge | - | 1 if API responding |
| ghost_active | gauge | - | 1 if engine active |
| ghost_error_count | gauge | - | In-memory error ring length |
| ghost_timestamp_seconds | gauge | - | Current server timestamp |
| ghost_decision_final_score | gauge | - | Fused final decision score (opt-in) |
| ghost_why_now_count | gauge | - | Count of top reasons included in signal card (opt-in) |
| ghost_llm_calls_total | counter | provider | Total LLM calls made (opt-in) |
| ghost_llm_decisions_total | counter | action | LLM decisions produced (opt-in) |
| ghost_llm_confidence | gauge | - | Last LLM decision confidence (opt-in) |

## PromQL Starters

```promql
rate(ghost_cockpit_snapshot_failures_total[5m])

histogram_quantile(0.5, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le))

histogram_quantile(0.95, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le))

# Average provider latency per provider

### Fusion and LLM examples

Final score (last value):

```promql

ghost_decision_final_score

```text

Why now reasons (last value):

```promql

ghost_why_now_count

```text

LLM call rate and recent confidence:

```promql

rate(ghost_llm_calls_total[5m])
ghost_llm_confidence

```text

sum by (provider)(rate(ghost_provider_fetch_seconds_sum[5m]))
  / sum by (provider)(rate(ghost_provider_fetch_seconds_count[5m]))

```text

## Alert Examples

```yaml

- alert: GhostSnapshotLatencyP95High


  expr: histogram_quantile(0.95, sum(rate(ghost_cockpit_snapshot_build_seconds_bucket[5m])) by (le)) > 2
  for: 10m
  labels:
    severity: warning
  annotations:
    summary: "Ghost snapshot latency P95 elevated"
    description: "P95 cockpit snapshot build time > 2s for 10m"

- alert: GhostSnapshotFailures


  expr: rate(ghost_cockpit_snapshot_failures_total[5m]) > 0
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "Ghost snapshot failures occurring"
    description: "Snapshot failure counter increasing continuously"

- alert: GhostInstanceDown


  expr: ghost_up == 0
  for: 2m
  labels:
    severity: critical
  annotations:
    summary: "Ghost instance down"
    description: "No metrics scrape reporting ghost_up=1 for 2 minutes"

```text

## UI Latency Badge

Client-side badge shows `tick: p50 / p95 ms` based on parsing time of last 30 SSE snapshot events.
Disable via `UI_LATENCY_BADGE=0`.

## Testing Failure Path

Force a snapshot failure (non-production):

```bash

SNAP_FORCE_FAIL=1 curl -sS <<<<<http://localhost:5000/api/cockpit>>>>>

```text

Check increment:

```bash

curl -s <<<<<http://localhost:5000/metrics>>>>> | grep ghost_cockpit_snapshot_failures_total

```text

## Multiprocess Note

If you scale to multiple workers, set `PROMETHEUS_MULTIPROC_DIR` and ensure the directory is empty on each restart.
Current deployment assumes single-process simplicity.

## Additions (WOLF-only app)

- New gauge `ghost_snapshot_asof`: last snapshot epoch from `/api/cockpit`.
  - Time since last snapshot (seconds): `(time() - ghost_snapshot_asof)`
  - Last seen timestamp window: `last_over_time(ghost_snapshot_asof[15m])`
- `/metrics` auto-detects Prometheus multiprocess mode via `PROMETHEUS_MULTIPROC_DIR`.
- Grafana example panel JSON for `ghost_snapshot_asof` lives at `docs/grafana/snapshot_asof_panel.json`.


---

## AI Agent Metrics (v2)

The ChatGPT Analyst agent exports additional metrics for monitoring decision quality and tool performance.

### Agent Metrics Table

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| ghost_ai_decision_confidence | gauge | - | Latest decision confidence (0-1) |
| ghost_ai_decisions_total | counter | action | Total decisions by action type (BUY/SELL/HOLD) |
| ghost_ai_tool_calls_total | counter | tool_name, result | Tool invocations (success/failure) |
| ghost_ai_tool_latency_seconds | histogram | tool_name | Tool call latency distribution |
| ghost_ai_decision_last_ts | gauge | - | Unix timestamp of last decision |

### Agent PromQL Examples

**Average confidence over last hour:**```promql

avg_over_time(ghost_ai_decision_confidence[1h])

```text**Decision rate by action:**```promql

rate(ghost_ai_decisions_total[5m])

```text**Tool success rate:**```promql

rate(ghost_ai_tool_calls_total{result="success"}[5m])
/
rate(ghost_ai_tool_calls_total[5m])

```text**Tool failure rate by tool:**```promql

rate(ghost_ai_tool_calls_total{result="failure"}[5m])

```text**Tool latency p95:**```promql

histogram_quantile(0.95,
  rate(ghost_ai_tool_latency_seconds_bucket[5m])
)

```text**Tool latency p95 by tool:**```promql

histogram_quantile(0.95,
  sum by (tool_name, le) (rate(ghost_ai_tool_latency_seconds_bucket[5m]))
)

```text**Time since last decision (seconds):**```promql

time() - ghost_ai_decision_last_ts

```text**Decisions per hour:**

```promql

rate(ghost_ai_decisions_total[1h]) * 3600

```text

### Agent Alert Rules

See `docs/alerts/agent_slo_rules.yml` for comprehensive alert definitions.

**Critical Alerts:**-**GhostAgentStale**: No decisions in 24+ hours → Agent may be down

- **GhostAgentToolFailures**: Tool failure rate >20% → Data provider issues
- **GhostAgentToolLatency**: Tool latency p95 >5s → Performance degradation


**Warning Alerts:**-**GhostAgentLowConfidence**: Avg confidence <50% for 30min → Market uncertainty

- **GhostAgentNotFetchingData**: No tool calls for 30min → Agent idle


**Info Alerts:**-**GhostAgentVeryLowConfidence**: Latest decision <30% confidence

- **GhostAgentHighDecisionRate**: >0.5 decisions/sec → Possible oscillation


### Agent Runbooks

**GhostAgentStale**- Agent stopped making decisions

1. Check health: `GET /agent/health`
2. Verify `OPENAI_API_KEY` is set in Railway
3. Check logs for errors: `railway logs --filter agent`
4. Restart service if needed: `railway restart`**GhostAgentLowConfidence**- Low confidence decisions

1. Review recent decisions: `GET /api/ai/decisions?hours=2`
2. Check market conditions (VIX, volatility)
3. Review decision rationale in agent responses
4. May be normal during high uncertainty periods**GhostAgentToolFailures**- High tool error rate

1. Check tool metrics: `GET /api/ai/monitor`
2. Verify API keys: `POLYGON_API_KEY`, `ALPHAVANTAGE_API_KEY`
3. Check provider status pages
4. Review specific errors in logs
5. Test tools manually: `GET /api/prices/WOLF`**GhostAgentToolLatency**- Slow tool responses

1. Check tool metrics by provider: `GET /api/ai/monitor`
2. Review Railway resource utilization (CPU, memory)
3. Check provider status for incidents
4. Consider increasing `HTTP_TIMEOUT_S` if appropriate
5. Check network latency to providers**GhostAgentNotFetchingData** - Agent idle

1. Check agent state: `GET /agent/state`
2. Verify `GHOST_AGENT_TICK` is 300s (5min)
3. Check `AI_ON=1` in environment variables
4. Review conversation history: `GET /agent/state`
5. Check portfolio has positions: `GET /api/cockpit`


### Grafana Dashboard

See `docs/grafana/agent_dashboard.json` for a complete monitoring dashboard with:

- Decision confidence gauge
- Action distribution pie chart
- Tool success rate by tool
- Tool latency heatmap
- Decisions timeline
- Tool calls rate graph

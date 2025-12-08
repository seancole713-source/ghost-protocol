# GHOST Reliability Checklist

**Version**: 1.0
**Date**: October 4, 2025
**Owner**: SRE Team / Platform Engineering
**Frequency**: Pre-release + Weekly

---

## Circuit Breakers & Backoff

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Exponential backoff configured | ⚠️ **BUGGY**| P1 |**GH-AUD-005**: Sticky backoff_factor |
| ✅ Backoff resets on success | ❌ **MISSING**| P1 | Line 2555: no `backoff_factor = 0` reset |
| ✅ Jitter added to retry delays | ❌**MISSING**| P1 | Causes thundering herd on recovery |
| ✅ Circuit breaker state exposed in /metrics | ✅**DONE**| P2 | Prometheus gauge `provider_breaker_state` |
| ✅ Max backoff capped to reasonable value | ✅**DONE**| P2 | `PROVIDER_BACKOFF_MAX_S=240` |
| ✅ Provider quorum logic handles partial failures | ✅**DONE**| P1 | Line 2680: 1-of-3 providers sufficient |**Action
Items**:

- [ ] Fix backoff reset in `_breaker_on_success()` (GH-AUD-005)
- [ ] Add 20% jitter: `backoff * (1 + random.uniform(-0.2, 0.2))`
- [ ] Add unit test: simulate 429 → success → 429 → verify backoff resets
- [ ] Document backoff behavior in README


---

## External Data Sources

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Reuters feed has degraded mode | ❌ **MISSING**| P1 |**GH-AUD-006**: DNS failure crashes loop |
| ✅ Yahoo Finance has fallback provider | ✅ **DONE**| P1 | Falls back to Polygon/AlphaVantage |
| ✅ Price staleness detection | ✅**DONE**| P1 | `_too_old()` checks timestamp |
| ✅ Cache hit rate monitored | ✅**DONE**| P2 | Prometheus counter `price_cache_hits` |
| ✅ News feed timeout configured | ⚠️**PARTIAL**| P2 | Per-request timeout, no outer timeout |
| ✅ DNS resolution errors handled gracefully | ❌**FAILS**| P1 | GH-AUD-006: propagates up, crashes feed |**Action
Items**:

- [ ] Wrap Reuters feed loop in outer try/except (GH-AUD-006)
- [ ] Return cached news with `"_degraded": true` flag on failure
- [ ] Add `NEWS_CACHE["last_refresh_ts"]` to track staleness
- [ ] Add alert: "News feed stale >30 min"
- [ ] Test degraded mode: kill DNS → verify UI shows cached news


---

## Health Checks & Liveness

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ /health endpoint responds <500ms | ✅ **FAST**| P0 | Line 4182: minimal logic |
| ✅ /health/detailed has timeout guard | ❌**MISSING**| P1 |**GH-AUD-011**: Can block >5s on DB lock |
| ✅ Railway healthcheck configured | ⚠️ **UNKNOWN**| P0 | Check Railway dashboard settings |
| ✅ Separate /ready and /live probes | ❌**MISSING**| P2 | `/health` serves both purposes |
| ✅ Startup probe allows warm-up time | ⚠️**UNKNOWN**| P2 | Railway setting |
| ✅ Health checks don't query DB by default | ⚠️**PARTIAL**| P1 | /health is fast, /health/detailed is slow |**Action
Items**:

- [ ] Add 3s timeout to `/health/detailed` DB queries (GH-AUD-011)
- [ ] Use `asyncio.wait_for(check_db(), timeout=3.0)`
- [ ] Create `/live` (process alive) and `/ready` (dependencies ready) endpoints
- [ ] Configure Railway to use `/live` for liveness, `/health` for readiness
- [ ] Test under DB write load: verify health doesn't block


---

## Background Tasks & Threads

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Autosave thread has error handling | ✅ **DONE**| P1 | Line 3550: try/except wrapper |
| ✅ Alert worker thread has error handling | ✅**DONE**| P1 | Line 3712: try/except wrapper |
| ✅ Scheduler thread has error handling | ✅**DONE**| P1 | Line 3829: try/except wrapper |
| ✅ SSE generators detect disconnects | ❌**MISSING**| P1 |**GH-AUD-004**: No client tracking/TTL |
| ✅ Thread crash doesn't kill main process | ✅ **DONE**| P0 | `daemon=True` prevents blocking |
| ✅ Thread health monitored | ⚠️**PARTIAL**| P2 | No heartbeat tracking |**Action Items**:

- [ ] Add `request.is_disconnected()` check to SSE generators (GH-AUD-004)
- [ ] Add TTL: close SSE after 5 minutes if no client activity
- [ ] Add Prometheus gauge: `background_thread_last_run_ts`
- [ ] Add alert: "Background thread not seen in 2 minutes"
- [ ] Test SSE cleanup: start stream, kill client, verify memory freed


---

## Database Resilience

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ SQLite WAL mode enabled | ⚠️ **UNKNOWN**| P1 | Check `db.py` pragma settings |
| ✅ Write-ahead log configured | ⚠️**UNKNOWN**| P1 | Reduces lock contention |
| ✅ DB connection pool used | ⚠️**N/A**| P2 | SQLite: one writer at a time |
| ✅ Transaction retries on SQLITE_BUSY | ⚠️**PARTIAL**| P1 | Some DB helpers have retry, not all |
| ✅ DB backups automated | ⚠️**MANUAL**| P2 | `scripts/railway_backup.py` exists |
| ✅ Portfolio persistence default mode safe | ⚠️**UNSAFE**| P1 | Default `"none"` → $0 on restart |**Action Items**:

- [ ] Enable WAL mode: `PRAGMA journal_mode=WAL;` in `db.py`
- [ ] Change default persistence to `"auto"` or `"always"`
- [ ] Add retry decorator for all DB writes (3 retries, exp backoff)
- [ ] Automate backups: Railway scheduled task daily
- [ ] Test restore: verify backup can boot cold server


---

## Duplicate Routes & State

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ No duplicate endpoint definitions | ❌ **DUPLICATE**| P1 |**GH-AUD-003**: Two `/api/cockpit/stream` |
| ✅ Legacy routes deprecated/removed | ⚠️ **LEGACY**| P2 |**GH-AUD-008**: `main.py` duplicates routes |
| ✅ State singleton enforced | ✅ **DONE**| P1 | Global dicts used consistently |
| ✅ No race conditions in state updates | ⚠️**UNKNOWN**| P2 | No formal concurrency audit |**Action Items**:

- [ ] Consolidate `/api/cockpit/stream` to single implementation (GH-AUD-003)
- [ ] Add collision detection test: `grep -n "^@app\\.get\\|^@APP\\.get" wolf_app.py | sort`
- [ ] Rename `main.py` to `main_DEPRECATED.py` (GH-AUD-008)
- [ ] Add threading lock for portfolio state mutations
- [ ] Run ThreadSanitizer or manual race condition review


---

## Monitoring & Alerting

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Prometheus metrics exposed | ✅ **DONE**| P1 | `/metrics` endpoint |
| ✅ Request rate monitored | ✅**DONE**| P1 | `http_requests_total` counter |
| ✅ Error rate monitored | ✅**DONE**| P1 | `http_errors_total` counter |
| ✅ Price fetch success rate monitored | ✅**DONE**| P1 | `price_fetches_total{result="success"}` |
| ✅ Circuit breaker state monitored | ✅**DONE**| P2 | `provider_breaker_state` gauge |
| ✅ Alerts configured in Railway | ⚠️**UNKNOWN**| P1 | Check Railway integrations |
| ✅ Alert runbook exists | ⚠️**MISSING**| P2 | Need ops runbook |**Action Items**:

- [ ] Configure Railway alert: CPU >80% for 5 minutes
- [ ] Configure Railway alert: Memory >90% for 2 minutes
- [ ] Configure Railway alert: Error rate >5% over 15 minutes
- [ ] Create `RUNBOOK.md` with response procedures
- [ ] Test alert delivery: Slack/PagerDuty integration


---

## Graceful Degradation

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ UI shows "stale data" indicators | ⚠️ **PARTIAL**| P2 | `_degraded` field exists but not always used |
| ✅ Portfolio survives provider outages | ✅**DONE**| P1 | Quorum logic works |
| ✅ News feed survives RSS failures | ⚠️**BUGGY**| P1 | GH-AUD-006: crashes instead of degrading |
| ✅ Telegram outage doesn't block operations | ✅**DONE**| P1 | Fire-and-forget HTTP calls |
| ✅ AI memory failure doesn't crash engine | ⚠️**UNKNOWN**| P2 | Need test: delete ai_memory.db → verify boot |**Action
Items**:

- [ ] Add `"_degraded": true` to all API responses with cached/stale data
- [ ] UI: show yellow banner when `_degraded` is true
- [ ] Test all failure modes: delete wolf.db, ai_memory.db, kill Redis
- [ ] Document degraded mode behavior in README
- [ ] Add E2E test: simulate all provider failures → verify UI still loads


---

## Deployment & Rollback

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Blue-green deployment available | ⚠️ **UNKNOWN**| P2 | Railway: check deployment strategy |
| ✅ Rollback tested within 5 minutes | ⚠️**UNKNOWN**| P1 | Need rollback drill |
| ✅ Database migrations reversible | ⚠️**N/A**| P2 | Schema changes rare |
| ✅ Zero-downtime deployment possible | ⚠️**UNKNOWN**| P2 | Railway setting |
| ✅ Health check prevents bad deploy | ⚠️**UNKNOWN**| P1 | Railway: verify health check gates |**Action Items**:

- [ ] Document Railway rollback procedure
- [ ] Test rollback: deploy bad version → rollback → verify works
- [ ] Add smoke test to CI/CD: deploy → curl /health → pass/fail
- [ ] Schedule quarterly deployment drill


---

## Capacity & Scaling

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Rate limiting protects from overload | ⚠️ **DISABLED**| P1 | `RATE_LIMIT_WRITE_RPM=0` by default |
| ✅ Connection pooling configured | ⚠️**N/A**| P2 | SQLite: single writer |
| ✅ Load testing performed | ⚠️**NEVER**| P2 | Consider Locust or K6 |
| ✅ Auto-scaling configured | ⚠️**UNKNOWN**| P3 | Railway: check scaling settings |
| ✅ Memory leaks identified | ⚠️**UNKNOWN**| P2 | Need long-running test (24h) |**Action Items**:

- [ ] Enable rate limiting in production: `RATE_LIMIT_WRITE_RPM=60`
- [ ] Run load test: 100 concurrent users for 1 hour
- [ ] Monitor memory usage over 24 hours (detect leaks)
- [ ] Document capacity limits: max users, max trades/day


---

## Testing & Validation

| Item | Status | Priority | Notes |
|------|--------|----------|-------|
| ✅ Integration tests cover happy paths | ✅ **DONE**| P1 | 80+ tests in `tests/` |
| ✅ Failure mode tests exist | ⚠️**PARTIAL**| P1 | Some tests like `test_backoff_429.py` |
| ✅ Chaos engineering tests | ❌**MISSING**| P3 | Random provider failures |
| ✅ E2E smoke test automated | ✅**DONE**| P1 | `ghost_smoke_test.sh` |
| ✅ Regression tests for P1 bugs | ⚠️**PARTIAL**| P1 | Need tests for GH-AUD-005, GH-AUD-006 |**Action Items**:

- [ ] Create `test_circuit_breaker_reset.py` (GH-AUD-005 regression test)
- [ ] Create `test_reuters_degraded_mode.py` (GH-AUD-006 regression test)
- [ ] Add chaos test: randomly fail 1 provider → verify quorum works
- [ ] Schedule weekly smoke test against staging


---

## Sign-Off

| Role | Name | Date | Signature |
|------|------|------|-----------|
| SRE Lead | TBD | YYYY-MM-DD | ____________ |
| Backend Lead | TBD | YYYY-MM-DD | ____________ |
| QA Lead | TBD | YYYY-MM-DD | ____________ |

**Next Review**: Weekly (Every Monday)

---

**Checklist Maintained By**: SRE Team
**Last Updated**: October 4, 2025
**Version**: 1.0
**Related Documents**: `GHOST_DEEP_AUDIT.md`, `UPGRADE_PLAN.md`

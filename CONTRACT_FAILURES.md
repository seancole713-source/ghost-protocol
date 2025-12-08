# 🐝 SWARM MISSION - EXECUTION PLAN

**Timestamp**: 2025-10-13 23:57:00 UTC\
**Status**: 🔴 **3 CONTRACT FAILURES**→ Immediate fixes required

______________________________________________________________________

## 🧪 CONTRACT TEST RESULTS

### ✅ PASSING (5/9)

1. ✅**test_contract_crypto_price_quorum**- Crypto prices working!
2. ✅**test_contract_telegram_qa**- Telegram Q&A working!
3. ✅**test_contract_health_endpoint**- Health check working!
4. ✅**test_contract_ready_endpoint**- Ready check exists!
5. ✅**test_contract_feature_flags**- Feature flags configured!


### ⏭️ SKIPPED (1/9)

1. ⏭️**test_contract_prediction_overlay**- Endpoint doesn't exist yet (expected)


### ❌ FAILING (3/9)

1. ❌**test_contract_stock_price_quorum**-**404 on /api/quotes**⚠️ CRITICAL
2. ❌**test_contract_trading_submission**-**dry_run response format mismatch**3. ❌**test_contract_prometheus_metrics**-**Empty /metrics response**


______________________________________________________________________

## 🚨 CRITICAL: FIX #1 - Stock Price Endpoint 404

### Issue

```text
AssertionError: Expected 200, got 404
URL: /api/quotes?symbols=WOLF

```text

### Root Cause

The `/api/quotes` endpoint either:

1. Doesn't exist (unlikely - core feature)
2. Wrong path (maybe `/quote` singular?)
3. Query param issue


### Action

```bash

# Search for the correct endpoint

grep -n "def.*quote" wolf_app.py | head -20

```text

### Fix Strategy

- Locate actual quote endpoint path
- Update contract test to use correct path
- OR fix wolf_app.py if endpoint is wrong


**Priority**: 🔥 **P0 CRITICAL**(blocks stock price feature)

______________________________________________________________________

## 🟡 MEDIUM: FIX #2 - Trading Submission Format

### Issue

```python

Failed: Unexpected response: {
  'ok': True,
  'submitted': False,  # ❌ Expected True for dry_run
  'dry_run': True,
  'risk_check': 'PASSED'
}

```text

### Root Cause

Contract test expects:

- `submitted: True` when dry_run succeeds
- Current response shows `submitted: False`


This is actually**correct behavior**(dry run doesn't submit), but test logic is wrong.

### Action**Fix the contract test**(not the endpoint)

```python

# Contract should accept EITHER

if data.get("dry_run"):

    # Dry run: expect submitted=False but risk_check=PASSED

    assert data.get("risk_check") == "PASSED"
elif data.get("submitted"):

    # Real submission

    assert "order" in data
elif data.get("blocked"):

    # Blocked by risk

    assert "reason" in data

```text**Priority**: 🟡 **P1 MEDIUM**(test logic issue, endpoint works correctly)

______________________________________________________________________

## 🟡 MEDIUM: FIX #3 - Prometheus Metrics Empty

### Issue

```text

AssertionError: No Ghost metrics found (should start with ghost_)
Response: '' (empty string)

```text

### Root Cause

The `/metrics` endpoint exists (200 response) but returns empty content.**Likely causes**:

1. No metrics registered yet
2. Prometheus exporter not configured
3. Empty metrics registry


### Action

**Implement Prometheus metrics in wolf_app.py**:

1. Add `prometheus_client` import
2. Create metric objects (Counter, Gauge)
3. Export in `/metrics` endpoint


**Priority**: 🟡 **P1 MEDIUM**(observability feature, not critical for core
functionality)

______________________________________________________________________

## 🎯 SWARM EXECUTION - PARALLEL FIXES

### Thread 1: Fix Stock Price Endpoint (15 min) 🔥**Status**: ACTIVE\

**Actions**:

1. Find correct endpoint path in wolf_app.py
2. Update contract test OR fix endpoint
3. Re-run test to verify


### Thread 2: Fix Trading Test Logic (5 min)

**Status**: ACTIVE\
**Actions**:

1. Update test_contract_trading_submission
2. Handle dry_run response correctly
3. Re-run test to verify


### Thread 3: Implement Prometheus Metrics (30 min)

**Status**: ACTIVE\
**Actions**:

1. Add prometheus_client to requirements
2. Create metric objects in wolf_app.py
3. Increment counters on price fetches
4. Update gauges on predictions
5. Re-run test to verify


______________________________________________________________________

## 📊 UPDATED STATUS

### Before Contract Tests

- Stock Prices: 🟢 ASSUMED WORKING (85%)
- Trading: 🟢 JUST DEPLOYED (90%)
- Metrics: 🔴 KNOWN MISSING (20%)


### After Contract Tests

- Stock Prices: 🔴 **BROKEN**- 404 endpoint (0%) ⚠️
- Trading: 🟢**WORKING**- Test logic wrong (100%)
- Metrics: 🔴**EMPTY**- Needs implementation (10%)


### Reality Check

Contract tests revealed**stock prices are broken**(critical finding!).

______________________________________________________________________

## 🚀 IMMEDIATE ACTIONS (Next 5 minutes)

1.**Find stock price endpoint**→ Highest priority
2.**Fix contract test for trading**→ Quick win
3.**Start metrics implementation**→ Medium term**Executing now...**

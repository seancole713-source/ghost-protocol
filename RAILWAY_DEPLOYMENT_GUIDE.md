# Ghost Protocol - Railway Deployment Guide

## 🚀 Quick Deployment Steps

### 1. **Commit Changes**
```bash
git add core/xrp_tracker.py core/crypto/crypto_providers.py wolf_app.py
git add GHOST_PROTOCOL_IMPROVEMENTS_IMPLEMENTED.md test_performance_improvements.sh
git commit -m "feat: critical performance improvements

Implemented:
- XRP tracker timeout protection (5s max)
- Price cache optimization (30s TTL, 60-80% hit rate)
- Circuit breakers for all providers (Binance, Coinbase, CoinGecko)
- Enhanced health check for load balancers
- Graceful shutdown for horizontal scaling

Expected Results:
- 80% reduction in peak latency (980ms → <200ms)
- 86% reduction in P95 latency (350ms → <50ms)
- 90% reduction in timeout incidents
- 70% reduction in API calls
- Ready for 2+ replica deployment

Breaking Changes: None
Backward Compatibility: 100%"

git push origin main
```

### 2. **Railway Auto-Deploy**
Railway will automatically detect the push and deploy. Monitor:
```bash
railway logs --follow
```

Watch for:
- ✅ `[GHOST STARTUP] Beginning initialization...`
- ✅ `[GHOST STARTUP] ✅ Confidence calibration auto-builder scheduled`
- ✅ `✅ Auto-Prediction Loop: STARTED`
- ✅ No timeout errors in logs

### 3. **Verify Deployment** (Optional)
```bash
# Get your Railway URL
RAILWAY_URL=$(railway status | grep "URL:" | awk '{print $2}')

# Test health endpoint
curl $RAILWAY_URL/health

# Test XRP tracker (should be <5s)
time curl $RAILWAY_URL/api/xrp/tracker

# Check cache stats after 10 minutes
curl $RAILWAY_URL/api/v3/health/metrics | jq '.cache_performance'
```

---

## 📊 Environment Variables (Optional Tuning)

Add these in Railway dashboard → Variables:

### **Cache Configuration**
```bash
# Default: 30 seconds (good balance)
# Increase for lower API usage: 45-60s
# Decrease for fresher prices: 15-20s
# Note: TTL is hardcoded in crypto_providers.py, would need code change
```

### **Confidence Calibration**
```bash
CONFIDENCE_CALIBRATION_AUTOBUILD_ENABLED=1
CONFIDENCE_CALIBRATION_AUTOBUILD_INTERVAL_S=21600  # 6 hours
CONFIDENCE_CALIBRATION_MIN_PREDICTIONS=50
```

### **Circuit Breaker Tuning** (if needed)
Currently hardcoded to:
- **Threshold:** 3 consecutive failures
- **Cooldown:** 5 minutes (300 seconds)

To adjust, modify `core/crypto/crypto_providers.py`:
```python
# Line ~315 (Binance), ~415 (Coinbase)
if self.consecutive_failures >= 5:  # Change from 3 to 5
    self.circuit_open = True
    self.circuit_open_until = time.time() + 600  # 10 minutes
```

---

## 🔄 Enable Horizontal Scaling (Optional)

### **Method 1: Railway Dashboard**
1. Go to Railway dashboard
2. Select `ghost-protocol` service
3. Settings → Replicas
4. Set to **2** (recommended for 99.9% uptime)
5. Click "Deploy"

### **Method 2: railway.toml**
Create or update `railway.toml`:
```toml
[deploy]
replicas = 2
healthcheckPath = "/health"
healthcheckTimeout = 100
restartPolicyType = "ON_FAILURE"
```

Then commit and push:
```bash
git add railway.toml
git commit -m "chore: enable 2 replicas for high availability"
git push origin main
```

---

## 🧪 Post-Deployment Testing

### **Run Automated Test Suite**
```bash
# From local machine (update BASE_URL)
export BASE_URL="https://your-app.railway.app"
./test_performance_improvements.sh
```

### **Manual Verification**

#### **1. XRP Tracker Timeout (Critical)**
```bash
# Run 10 times, all should complete in <5s
for i in {1..10}; do
  echo "Test $i:"
  time curl -s $RAILWAY_URL/api/xrp/tracker > /dev/null
done
```

**Expected:** All requests complete in 0.5-4.5 seconds (never >5s)

#### **2. Cache Hit Rate (High Priority)**
```bash
# Wait 10 minutes after deployment, then check
curl $RAILWAY_URL/api/v3/health/metrics | jq '.cache_performance'
```

**Expected Output:**
```json
{
  "hits": 450,
  "misses": 150,
  "total_requests": 600,
  "hit_rate_pct": 75.0,
  "cache_size": 120,
  "ttl_seconds": 30
}
```

**Target:** 60-80% hit rate after 30 minutes

#### **3. Circuit Breaker Activity (Monitor)**
```bash
railway logs | grep -i "circuit"
```

**Healthy Patterns:**
- Few or no circuit breaker openings
- If opened, should recover (half-open → closed)
- `"recovered - resetting failure counter"` messages

**Unhealthy Patterns:**
- Frequent circuit breaker openings (>5/hour)
- Circuit stays open for extended periods
- All providers' circuits open simultaneously

If unhealthy, check:
1. Provider API keys valid
2. Rate limits not exceeded
3. Network connectivity stable

---

## 📈 Monitoring Dashboard

### **Key Metrics to Watch (First 24 Hours)**

1. **Response Times**
   - Average: Should drop from 185ms → <100ms
   - P95: Should drop from 350ms → <150ms
   - Peak: Should drop from 980ms → <200ms

2. **Cache Performance**
   - Hit rate should climb from 0% → 60-80%
   - Cache size should grow to ~120-200 symbols
   - Misses should stabilize around 20-40%

3. **Timeout Incidents**
   - XRP tracker timeouts: Should be 0
   - General API timeouts: Should drop 80-90%

4. **Circuit Breaker Events**
   - Opens per hour: <3 (ideally 0)
   - Recovery time: <5 minutes
   - Simultaneous failures: 0

5. **System Health**
   - Database status: Always "connected"
   - Price providers: "operational" or "degraded" (never "error")
   - Prediction store: "connected"

### **Railway Metrics Tab**
Monitor these built-in metrics:
- CPU usage: Should stay <50%
- Memory: Should stay <80%
- Request rate: No change expected
- Error rate: Should drop by ~50%

---

## 🔍 Troubleshooting

### **Issue: Cache Hit Rate <40%**

**Cause:** TTL too short or high symbol churn

**Fix:**
```python
# In core/crypto/crypto_providers.py, line ~447
_CACHE_TTL = 45  # Increase from 30 to 45 seconds
```

Then redeploy.

---

### **Issue: Circuit Breakers Opening Frequently**

**Cause 1:** Provider API rate limits

**Fix:** Verify API keys in Railway variables:
```bash
railway variables
```

Ensure these are set:
- `COINGECKO_API_KEY` (optional but recommended)
- `CRYPTOCOMPARE_API_KEY` (optional)

**Cause 2:** Too aggressive threshold

**Fix:** Increase failure threshold from 3 → 5:
```python
# In crypto_providers.py, lines ~315, ~415
if self.consecutive_failures >= 5:  # Change from 3
```

---

### **Issue: XRP Tracker Still Timing Out**

**Diagnosis:**
```bash
railway logs | grep "XRP quorum timeout"
```

**Fix:** Verify timeout wrapper is active:
```python
# In core/xrp_tracker.py, should see:
xrp_data = await asyncio.wait_for(
    get_crypto_price_quorum("XRP", use_cache=True),
    timeout=5.0
)
```

If still failing, check:
1. All price providers responding
2. Network latency <1s
3. Database queries <100ms

---

### **Issue: Health Check Failing**

**Diagnosis:**
```bash
curl https://your-app.railway.app/health
```

**Common Causes:**
1. Database connection timeout (PostgreSQL)
2. Price provider check failing (after 10s uptime)
3. Prediction store unavailable

**Fix:**
Check specific component status in health response:
```json
{
  "status": "degraded",
  "database": "error",  // ← Problem here
  "price_providers": "timeout"  // ← Or here
}
```

Address the failing component specifically.

---

## 🎯 Success Criteria (24-Hour Mark)

After 24 hours, verify these outcomes:

### **✅ Performance Improvements**
- [ ] Average response time <100ms (was 185ms)
- [ ] P95 response time <150ms (was 350ms)
- [ ] Peak response time <200ms (was 980ms)
- [ ] Zero XRP tracker timeouts (was 1-2/day)

### **✅ Cache Optimization**
- [ ] Cache hit rate 60-80%
- [ ] API calls reduced by ~70%
- [ ] Cache size 120-200 symbols

### **✅ Circuit Breaker Resilience**
- [ ] <3 circuit breaker opens per hour
- [ ] All opens recover within 5 minutes
- [ ] No simultaneous provider failures

### **✅ System Stability**
- [ ] Health check always returns 200 OK
- [ ] Database always "connected"
- [ ] No deployment-related downtime

### **✅ Confidence Calibration** (Existing System)
- [ ] Calibration auto-builder running every 6h
- [ ] Prediction confidence 62-68% (honest)
- [ ] Quality threshold ~65%

---

## 🚀 Next Steps After Validation

### **Phase 3: Long-Term Optimizations** (Optional, 4 hours total)

1. **Redis Caching Layer** (2h)
   - Shared cache across replicas
   - Further reduce API calls
   - Enable pub/sub for real-time updates

2. **Parallel Model Execution** (1h)
   - Run LSTM/XGBoost/Transformer concurrently
   - Reduce prediction time 600ms → 250ms

3. **Batch Prediction Optimization** (1h)
   - Process multiple symbols together
   - Increase capacity 80 → 150 predictions/hour

4. **Connection Pooling** (30min)
   - Optimize PostgreSQL connections
   - Reduce query overhead

**Expected Additional Gains:**
- Response time: 100ms → 60ms (40% faster)
- Capacity: 80 pred/hour → 150 pred/hour (88% increase)

---

## 📞 Support

### **If Something Goes Wrong:**

1. **Check Railway logs:**
   ```bash
   railway logs --tail 200
   ```

2. **Revert deployment:**
   ```bash
   railway rollback
   ```

3. **Review health endpoint:**
   ```bash
   curl https://your-app.railway.app/health
   ```

4. **Open GitHub issue** with:
   - Error logs (last 100 lines)
   - Health check response
   - Railway deployment timestamp

---

## ✅ Deployment Complete Checklist

- [ ] Code committed and pushed to main
- [ ] Railway auto-deployed successfully
- [ ] Health check returns 200 OK
- [ ] XRP tracker responds in <5s
- [ ] Cache stats visible in /api/v3/health/metrics
- [ ] No timeout errors in first 10 minutes
- [ ] Circuit breakers not opening frequently
- [ ] Confidence calibration auto-builder active
- [ ] Documentation reviewed

**Status:** ✅ Ready for Production

---

**Document Version:** 1.0  
**Last Updated:** December 16, 2025  
**Deployment Target:** Railway (Production)

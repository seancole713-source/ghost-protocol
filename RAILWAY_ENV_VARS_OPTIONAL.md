# Optional Railway Environment Variables

All these settings now have **sensible defaults** and work out-of-the-box. Only add them if you need to tune performance or behavior.

## 🔧 PostgreSQL Connection Pool

Control database connection pooling for concurrent load:

```bash
POSTGRES_POOL_MIN=10           # Minimum connections (default: 10)
POSTGRES_POOL_MAX=50           # Maximum connections (default: 50)
POSTGRES_POOL_TIMEOUT_S=10     # Connection timeout seconds (default: 10)
```

**When to adjust:**
- Increase `POSTGRES_POOL_MAX` to 75-100 if you see "connection pool exhausted" errors
- Decrease to 25 if Railway reports high memory usage from idle connections

---

## 🤖 Auto-Prediction Performance Tuning

Control prediction loop behavior and resource usage:

```bash
# Deduplication
PREDICTION_DEDUP_WINDOW_S=300  # Skip predictions within 5 min (default: 300)

# Prediction intervals
AUTO_PREDICT_MARKET_INTERVAL_S=3600      # Market hours cycle: 60 min (default: 3600)
AUTO_PREDICT_OFF_HOURS_INTERVAL_S=7200   # Off hours cycle: 120 min (default: 7200)

# Concurrency & delays
AUTO_PREDICT_BATCH_SIZE=2           # Predictions per batch (default: 2)
AUTO_PREDICT_BATCH_DELAY_S=10       # Seconds between batches (default: 10)
AUTO_PREDICT_DELAY_S=2.0            # Seconds between predictions (default: 2.0)
AUTO_PREDICT_MAX_WORKERS=10         # Worker pool size (default: 10)
```

**When to adjust:**
- **Speed up predictions:** Decrease `AUTO_PREDICT_BATCH_DELAY_S` to 5 (but watch CPU)
- **Reduce load:** Increase `PREDICTION_DEDUP_WINDOW_S` to 600 (10 minutes)
- **More concurrency:** Increase `AUTO_PREDICT_BATCH_SIZE` to 3-4 (but watch pool exhaustion)

---

## 🌐 HTTP Connection Pool

Control outbound HTTP connections (yfinance, external APIs):

```bash
HTTP_POOL_SIZE=20              # HTTP connection pool size (default: 20, was 10)
HTTP_POOL_RETRIES=2            # Retry failed requests (default: 2)
HTTP_TIMEOUT_S=8               # Request timeout seconds (default: 8)
```

**When to adjust:**
- Increase `HTTP_POOL_SIZE` to 30-40 if `/watchlist/user` is slow (many yfinance calls)
- Decrease to 10 if you see "too many open files" errors

---

## 🪙 Coinbase Pro Configuration

Control Coinbase Pro API integration (RSI/trend data source):

```bash
COINBASE_PRO_ENABLED=1         # Enable Coinbase Pro (default: 1)
COINBASE_PRO_TIMEOUT_S=5.0     # API timeout seconds (default: 5.0)
COINBASE_PRO_BASE_URL=https://api.exchange.coinbase.com  # API base URL
```

**When to adjust:**
- Set `COINBASE_PRO_ENABLED=0` to temporarily disable if Coinbase has issues
- Increase `COINBASE_PRO_TIMEOUT_S` to 10 if seeing timeout errors

---

## 💾 Cache TTL Settings

Control caching for high-traffic endpoints to reduce database load:

```bash
HUNTER_FEED_CACHE_TTL=30       # Hunter feed cache seconds (default: 30)
WATCHLIST_CACHE_TTL=60         # Watchlist cache seconds (default: 60)
VIP_SNAPSHOT_CACHE_TTL=30      # VIP snapshot cache seconds (default: 30)
```

**When to adjust:**
- **Faster updates:** Decrease `WATCHLIST_CACHE_TTL` to 30 (but increases DB load)
- **Reduce DB load:** Increase to 120-300 (but data may be stale)

---

## 📊 Current Railway Settings (Defaults)

Your current setup uses all defaults. You **don't need to add these** unless tuning:

| Variable | Default | Your Status |
|----------|---------|-------------|
| `POSTGRES_POOL_MIN` | 10 | ✅ Using default |
| `POSTGRES_POOL_MAX` | 50 | ✅ Using default |
| `HTTP_POOL_SIZE` | 20 | ✅ Using new default |
| `PREDICTION_DEDUP_WINDOW_S` | 300 | ✅ Using default |
| `AUTO_PREDICT_BATCH_SIZE` | 2 | ✅ Using default |
| `AUTO_PREDICT_BATCH_DELAY_S` | 10 | ✅ Using default |
| `COINBASE_PRO_ENABLED` | 1 | ✅ Using default |
| All cache TTLs | 30-60s | ✅ Using defaults |

---

## 🎯 Recommended Actions

**Immediate (High Priority):**
1. ✅ **No action needed** - All settings have good defaults
2. ❌ **Still missing:** `CRYPTOPANIC_API_KEY` (see main env vars doc)

**Optional (If Issues Arise):**
- If "connection pool exhausted": Set `POSTGRES_POOL_MAX=75`
- If predictions too slow: Set `AUTO_PREDICT_BATCH_DELAY_S=5`
- If `/watchlist/user` slow: Set `HTTP_POOL_SIZE=30`

---

## 📝 How to Add Variables on Railway

1. Go to Railway Dashboard → Your project → `ghost-protocol`
2. Click **Variables** tab
3. Click **+ New Variable**
4. Add name (e.g., `POSTGRES_POOL_MAX`) and value (e.g., `75`)
5. Click **Add** → Service will automatically redeploy

**Note:** Changes take effect immediately on next deployment (automatic when you add variable).

---

## 🔍 Monitoring After Changes

After adding any variables, check Railway logs for:

```
Initializing Postgres connection pool (attempt 1/3)...
Pool config: minconn=10, maxconn=50, timeout=10s
```

This confirms your settings are being used. If you don't see your values, check:
1. Variable name spelling (case-sensitive)
2. Variable is added to correct service (`ghost-protocol` not another)
3. Redeploy triggered (Railway usually auto-redeploys on variable change)

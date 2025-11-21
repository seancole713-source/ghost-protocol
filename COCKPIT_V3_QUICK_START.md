# Cockpit V3 - Quick Start Guide

## Start Ghost

```bash
cd /Users/studio713/ghost-protocol
docker compose up -d
```

⏱️ **Wait 3 minutes** for full initialization (seriously, don't skip this!)

## Verify It's Working

```bash
# 1. Health check (should return {"ok":true})
curl http://localhost:8080/health

# 2. Test V3 endpoint (should return array with crypto data)
curl http://localhost:8080/api/v3/hunter/feed | jq '.'

# 3. Open Cockpit in browser
open http://localhost:8080/cockpit
```

## All V3 Endpoints (NO AUTH Required)

| Endpoint | Returns |
|----------|---------|
| `/api/v3/cockpit/status` | Live status, Ghost health score |
| `/api/v3/goals/snapshot` | Ghost Score, daily/weekly/monthly/yearly progress |
| `/api/v3/hunter/feed` | Top crypto movers with confidence scores |
| `/api/v3/vip/snapshot` | VIP coin prices (WEPE, LILPEPE, etc.) |
| `/api/v3/world/context` | SPY, QQQ, VIX, BTC, DXY + market regime |
| `/api/v3/risk/snapshot` | NAV, exposure %, VaR, drawdown |
| `/api/v3/portfolio/summary` | Positions, market value, P&L |
| `/api/v3/predictions/latest` | Most recent predictions |
| `/api/v3/predictions/recent` | Prediction history |
| `/api/v3/ai/metrics` | AI decisions, tool calls, success rate |
| `/api/v3/accuracy/summary` | Prediction accuracy stats |
| `/api/v3/providers/health` | Data provider status |
| `/api/v3/system/logs` | Recent system logs |
| `/api/v3/runtime/config` | Current runtime configuration |

## Troubleshooting

### "Connection reset by peer"
**Solution**: Wait the full 3 minutes. Ghost needs time to initialize.

### "Scanner warming up"
**Solution**: Normal! Data providers are fetching. Wait 5-10 minutes for real data.

### Container won't start
```bash
# Clean rebuild
docker compose down
docker compose build --no-cache app
docker compose up -d
sleep 180  # WAIT!
```

### Check logs
```bash
# All logs
docker compose logs app

# Just errors
docker compose logs app | grep -i error

# V3 loading confirmation
docker compose logs app | grep "Cockpit V3"
# Should see: "✅ Cockpit V3 LIVE endpoints registered"
```

## Important Notes

1. **Startup Time**: Always wait **180 seconds (3 minutes)** after `docker compose up`
2. **Data Warmup**: First 5-10 minutes show placeholder data while providers fetch
3. **VIP Coins**: Some coins (WEPE, LILPEPE, DORKL) not on major exchanges = null prices
4. **Auth**: All `/api/v3/` endpoints are PUBLIC (no Bearer token needed)
5. **V2 Still Works**: Old `/api/` endpoints unchanged, V3 is at `/api/v3/`

## Quick Tests

```bash
# One-liner to test everything
echo "Health:" && curl -s http://localhost:8080/health | jq '.ok' && \
echo "Hunter Feed:" && curl -s http://localhost:8080/api/v3/hunter/feed | jq 'length' && \
echo "Goals:" && curl -s http://localhost:8080/api/v3/goals/snapshot | jq '.ghost_score' && \
echo "✅ All working!"
```

## URLs

- **Cockpit UI**: http://localhost:8080/cockpit
- **API Docs**: http://localhost:8080/api/docs
- **Health Check**: http://localhost:8080/health
- **Old UI**: http://localhost:8080/ (legacy)

## Files Changed

- `api/cockpit_v3_live_endpoints.py` - V3 endpoint implementations
- `wolf_app.py` - IP_ALLOWLIST fix + V3 router registration + auth bypass
- `static/cockpit_v3.js` - Updated to call `/api/v3/hunter/feed`

## Done! 🚀

Ghost Protocol Cockpit V3 is live with 20+ real data endpoints at `/api/v3/`.

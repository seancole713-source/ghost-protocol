#!/bin/bash
# Quick Reference: Copy/paste commands for deployment verification

# ============================================
# STEP 1: Push to Railway (from host machine)
# ============================================
cd /path/to/ghost-cockpit
git pull origin master  # Get commit 2d772ed
git push railway main   # Deploy to Railway

# ============================================
# STEP 2: Wait for Railway deployment
# ============================================
# Dashboard → Wait for "Healthcheck succeeded!" (~2-3 minutes)

# ============================================
# STEP 3: Export environment variables
# ============================================
export GHOST_BASE_URL="https://ghost-sniper-bot-seancole713-production.up.railway.app"
export GHOST_API_TOKEN="edaa4eac-6455-4693-a745-142cb6deef03"

# ============================================
# STEP 4: Run automated smoke tests
# ============================================
bash /app/deployment_smoke_test.sh

# ============================================
# STEP 5: Manual endpoint tests (if needed)
# ============================================

# Status
curl -s "$GHOST_BASE_URL/api/status" | python -m json.tool

# Tick (run twice to see increment)
curl -s "$GHOST_BASE_URL/api/tick" | python -m json.tool
sleep 3
curl -s "$GHOST_BASE_URL/api/tick" | python -m json.tool

# Regime
curl -s "$GHOST_BASE_URL/api/regime/current" | python -m json.tool

# Diagnostics (AAPL routing fix verification)
curl -s "$GHOST_BASE_URL/api/price/diagnostics?symbol=AAPL" | python -m json.tool

# Cache purge
curl -s -H "Authorization: Bearer $GHOST_API_TOKEN" \
     -H "Content-Type: application/json" \
     -X POST \
     -d '{"patterns":["price:AAPL","diagnostics:*"]}' \
     "$GHOST_BASE_URL/api/cache/purge" | python -m json.tool

# Movers health
curl -s "$GHOST_BASE_URL/api/scan/health" | python -m json.tool

# OpenAPI schema check
curl -s "$GHOST_BASE_URL/openapi.json" | python -m json.tool | grep -E "(regime|tick|purge|movers)"

# ============================================
# STEP 6: Browser verification (SSE)
# ============================================
# Open in browser:
# https://ghost-sniper-bot-seancole713-production.up.railway.app/api/cockpit/stream
#
# Expected: Continuous stream of events
#   event: status
#   event: ping
#   event: snapshot

# ============================================
# STEP 7: Monitor Railway logs (5 minutes)
# ============================================
# Dashboard → Logs → HTTP Logs
# Check for: 0×499, 0×502 errors
# Any 500s should be JSON: {"error": "internal_error"}

# Alternative: Via Railway CLI
railway logs --follow | grep -E '(499|502|500)'

# ============================================
# SUCCESS CRITERIA
# ============================================
# ✅ All endpoints return 200 (not 404)
# ✅ /api/tick counter increments
# ✅ AAPL diagnostics shows AAPL price (not WOLF)
# ✅ Cache purge returns {"ok": true}
# ✅ SSE stream continuously emits events
# ✅ No 499 or 502 errors in logs for 5 minutes
# ✅ Average response time < 3000ms

# ============================================
# ROLLBACK (if needed)
# ============================================
# Railway Dashboard → Deployments → Select previous → Redeploy
# OR
railway rollback

# ============================================
# REPORT FAILURES
# ============================================
# If any test fails, provide:
# 1. Which endpoint failed
# 2. HTTP status code received
# 3. Response body (first 500 chars)
# 4. Expected vs actual behavior

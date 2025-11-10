#!/bin/bash
# Railway Deployment Verification Script (enhanced)
set -euo pipefail

echo "🚀 GHOST Railway Deployment Verification"
echo "=========================================="
echo ""

RAILWAY_URL="https://web-production-8e9a0.up.railway.app"
TIMESTAMP=$(date +%s)

have_jq() { command -v jq >/dev/null 2>&1; }

json_key() {
  # json_key <json> <jq_expr> -> value or empty
  if have_jq; then
    echo "$1" | jq -r "$2" 2>/dev/null || true
  else
    # Fallback using Python
    python3 - "$2" <<'PY'
import json,sys
expr = sys.argv[1]
data = sys.stdin.read()
try:
    j = json.loads(data)
    # limited extractor for common keys only
    if expr == '.ok':
        print('true' if j.get('ok') else 'false')
    elif expr == '.components.ai_memory':
        c = j.get('components',{}).get('ai_memory')
        print(c if c is not None else '')
    elif expr == '.paths | length':
        print(len(j.get('paths',{})))
    else:
        print('')
except Exception:
    print('')
PY
  fi
}

echo "1️⃣ Checking Health Endpoint..."
health_status=$(curl -s -o /dev/null -w "%{http_code}" "${RAILWAY_URL}/health")
if [ "$health_status" = "200" ]; then
    echo "   ✅ Health check passed (HTTP $health_status)"
    if have_jq; then curl -s "${RAILWAY_URL}/health" | jq -r '.status // "OK"' || true; fi
else
    echo "   ❌ Health check failed (HTTP $health_status)"
fi
echo ""

echo "2️⃣ Checking Detailed Health..."
detailed=$(curl -s "${RAILWAY_URL}/health/detailed")
ok=$(json_key "$detailed" '.ok')
if [ "$ok" = "true" ]; then
    echo "   ✅ Detailed health check passed"
    am=$(json_key "$detailed" '.components.ai_memory')
    [ -n "$am" ] && echo "   ai_memory: $am"
else
    echo "   ⚠️  Detailed health has issues"
    if have_jq; then echo "$detailed" | jq -r '.issues // []' || true; fi
fi
echo ""

echo "3️⃣ OpenAPI route count..."
openapi=$(curl -s "${RAILWAY_URL}/openapi.json" || true)
routes=$(json_key "$openapi" '.paths | length')
if [ -n "$routes" ]; then
  echo "   ℹ️  Routes reported by prod: $routes"
else
  echo "   ⚠️  Unable to fetch OpenAPI JSON"
fi
echo ""

echo "4️⃣ Checking Cockpit API..."
cockpit_status=$(curl -s -o /dev/null -w "%{http_code}" "${RAILWAY_URL}/api/cockpit")
if [ "$cockpit_status" = "200" ]; then
    echo "   ✅ Cockpit API responding (HTTP $cockpit_status)"
else
    echo "   ❌ Cockpit API failed (HTTP $cockpit_status)"
fi
echo ""

echo "5️⃣ Checking Static Assets (Cache Busting)..."
index_status=$(curl -s -o /dev/null -w "%{http_code}" "${RAILWAY_URL}/?v=${TIMESTAMP}")
if [ "$index_status" = "200" ]; then
    echo "   ✅ Index page loaded (HTTP $index_status)"
else
    echo "   ❌ Index page failed (HTTP $index_status)"
fi
echo ""

echo "6️⃣ Checking Metrics..."
metrics=$(curl -s "${RAILWAY_URL}/metrics" | head -20 || true)
if echo "$metrics" | grep -q "ghost_up"; then
    echo "   ✅ Prometheus metrics available"
    echo "$metrics" | grep "ghost_up" | head -3
else
    echo "   ⚠️  Metrics not found or malformed"
fi
echo ""

echo "7️⃣ Probing critical endpoints..."
probe() {
  path="$1"; expect="$2"
  code=$(curl -s -o /dev/null -w "%{http_code}" "${RAILWAY_URL}${path}" || echo "000")
  printf "   %-28s -> HTTP %s" "$path" "$code"
  if [ "$expect" != "" ]; then
    if [ "$code" = "$expect" ]; then echo "  ✅"; else echo "  ❌ (expected $expect)"; fi
  else
    echo
  fi
}

probe "/api/news" 200
probe "/api/news/recent" 200
probe "/api/news/sentiment/WOLF" 200
probe "/api/agent/decisions" 200
probe "/api/agent/stats" 200
probe "/api/portfolio" 200
probe "/api/snapshot" 200
probe "/api/stage2/forecasts" 200
probe "/api/stage1/world" 200
echo ""

echo "8️⃣ UI Panel keywords present..."
ui_content=$(curl -s "${RAILWAY_URL}/" 2>/dev/null || true)
if echo "$ui_content" | grep -q "Intelligence Engine\|Market Pulse\|Predictive Analytics"; then
    echo "   ✅ UI panel names found"
else
    echo "   ℹ️  UI bundle may not be present (this is OK if serving API-only)"
fi
echo ""

echo "=========================================="
echo "📊 DEPLOYMENT STATUS SUMMARY"
echo "=========================================="

passed=0
total=8
[ "$health_status" = "200" ] && ((passed++))
[ "$ok" = "true" ] && ((passed++))
[ -n "${routes}" ] && ((passed++))
[ "$cockpit_status" = "200" ] && ((passed++))
[ "$index_status" = "200" ] && ((passed++))
echo "$metrics" | grep -q "ghost_up" && ((passed++))
# Consider news feed as critical signal
news_code=$(curl -s -o /dev/null -w "%{http_code}" "${RAILWAY_URL}/api/news")
[ "$news_code" = "200" ] && ((passed++))
# Portfolio basic signal
portfolio_code=$(curl -s -o /dev/null -w "%{http_code}" "${RAILWAY_URL}/api/portfolio")
[ "$portfolio_code" = "200" ] && ((passed++))

echo ""
echo "Result: $passed/$total checks passed"
if [ -n "$routes" ]; then
  echo "Routes (prod): $routes"
fi
echo ""

if [ "$passed" -ge 6 ]; then
    echo "✅ Deployment verified successfully!"
    echo ""
    echo "🌐 Your GHOST instance is live at:"
    echo "   $RAILWAY_URL"
    echo ""
    echo "🔗 Quick Links:"
    echo "   Dashboard: ${RAILWAY_URL}/api/cockpit"
    echo "   Health:    ${RAILWAY_URL}/health/detailed"
    echo "   Metrics:   ${RAILWAY_URL}/metrics"
    exit 0
else
    echo "❌ Deployment verification failed"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "   1. Check Railway logs: railway logs"
    echo "   2. Verify environment variables"
    echo "   3. Restart service: railway restart"
    echo "   4. Force redeploy: railway up --detach"
    exit 1
fi

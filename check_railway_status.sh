#!/bin/bash
# Quick Railway deployment verification script

BASE="https://web-production-8e9a0.up.railway.app"

echo "🔍 RAILWAY DEPLOYMENT STATUS CHECK"
echo "==================================="
echo ""

echo "📍 Latest local commit:"
git log --oneline -1
echo ""

echo "🏥 Production health check:"
curl -s "$BASE/health" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'  Status: ✅ UP' if d.get('ok') else '  Status: ❌ DOWN'); print(f'  Timestamp: {d.get(\"ts\", \"unknown\")}')" 2>/dev/null || echo "  Status: ❌ UNREACHABLE"
echo ""

echo "📊 OpenAPI schema analysis:"
curl -s "$BASE/openapi.json" | python3 -c "
import json, sys
d = json.load(sys.stdin)
paths = list(d['paths'].keys())
news = [p for p in paths if 'news' in p.lower()]
agent = [p for p in paths if 'agent' in p.lower()]
snapshot = [p for p in paths if 'snapshot' in p.lower()]

print(f'  Total routes: {len(paths)}')
print(f'  News routes: {len(news)}')
print(f'  Agent routes: {len(agent)}')
print(f'  Snapshot routes: {len(snapshot)}')
print()
print('  News route details:')
for p in news[:10]:
    print(f'    - {p}')
" 2>/dev/null || echo "  ❌ Failed to fetch OpenAPI schema"
echo ""

echo "🧪 Endpoint tests:"
echo "  /api/news:"
HTTP_CODE=$(curl -s -o /tmp/news_test.json -w "%{http_code}" "$BASE/api/news?limit=2")
if [ "$HTTP_CODE" = "200" ]; then
    echo "    ✅ HTTP $HTTP_CODE"
    python3 -c "import json; d=json.load(open('/tmp/news_test.json')); print(f'    📰 {d.get(\"count\", 0)} items')"
else
    echo "    ❌ HTTP $HTTP_CODE"
fi

echo "  /api/news/recent:"
HTTP_CODE=$(curl -s -o /tmp/news_recent_test.json -w "%{http_code}" "$BASE/api/news/recent?minutes=120")
if [ "$HTTP_CODE" = "200" ]; then
    echo "    ✅ HTTP $HTTP_CODE"
    python3 -c "import json; d=json.load(open('/tmp/news_recent_test.json')); print(f'    📰 {d.get(\"count\", 0)} items')"
else
    echo "    ❌ HTTP $HTTP_CODE"
fi

echo "  /debug/router_status:"
HTTP_CODE=$(curl -s -o /tmp/debug_test.json -w "%{http_code}" "$BASE/debug/router_status")
if [ "$HTTP_CODE" = "200" ]; then
    echo "    ✅ HTTP $HTTP_CODE"
    python3 -c "import json; d=json.load(open('/tmp/debug_test.json')); print(f'    🔧 Total routes: {d.get(\"total_routes\", 0)}'); print(f'    📰 News routes: {len(d.get(\"news_routes\", []))}')"
else
    echo "    ❌ HTTP $HTTP_CODE"
fi

echo ""
echo "🎯 EXPECTED VALUES AFTER SUCCESSFUL DEPLOY:"
echo "   - Total routes: 259 (currently seeing 231)"
echo "   - News routes: 5 (currently seeing 1 old route)"
echo "   - /api/news: HTTP 200"
echo "   - /api/news/recent: HTTP 200"
echo "   - /debug/router_status: HTTP 200"
echo ""
echo "💡 IF STILL FAILING:"
echo "   1. Go to Railway Dashboard → Deployments"
echo "   2. Find commit 0788e72 (requirements fix)"
echo "   3. Click 'Rebuild' with cache disabled"
echo "   4. Watch build logs for errors"
echo ""

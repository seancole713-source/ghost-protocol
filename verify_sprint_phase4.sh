#!/bin/bash
# Sprint Phase 4 Verification Script
# Run after server restart to verify all fixes

echo "🔍 Ghost Sprint Phase 4 - Verification Script"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

BASE_URL="${GHOST_URL:-http://localhost:5000}"

echo "📡 Testing server connectivity..."
if curl -s -f "$BASE_URL/health" > /dev/null; then
    echo -e "${GREEN}✅ Server is responding${NC}"
else
    echo -e "${RED}❌ Server is not responding${NC}"
    exit 1
fi

echo ""
echo "🔧 Checking scheduler configuration..."
SCHED_ENABLED=$(curl -s "$BASE_URL/api/config" | python3 -c "import sys,json; print(json.load(sys.stdin).get('alerts',{}).get('schedule_open_close',False))" 2>/dev/null)
if [ "$SCHED_ENABLED" = "True" ]; then
    echo -e "${GREEN}✅ Market open/close scheduler is ENABLED${NC}"
else
    echo -e "${RED}❌ Scheduler is DISABLED - restart server with ALERT_SCHEDULE_OPEN_CLOSE=1${NC}"
fi

echo ""
echo "📊 Checking metrics endpoint..."
METRICS_SIZE=$(curl -s "$BASE_URL/metrics" | wc -c)
if [ "$METRICS_SIZE" -gt 100 ]; then
    echo -e "${GREEN}✅ Metrics endpoint returning data ($METRICS_SIZE bytes)${NC}"
else
    echo -e "${YELLOW}⚠️  Metrics endpoint empty ($METRICS_SIZE bytes) - may need eager initialization${NC}"
    echo "   See: PROMETHEUS_METRICS_DEBUG.md"
fi

echo ""
echo "📱 Testing Telegram configuration..."
TELEGRAM_TEST=$(curl -s -X POST "$BASE_URL/api/telegram/test?send=false" | python3 -c "import sys,json; d=json.load(sys.stdin); print('ok' if d.get('can_send') else 'error')" 2>/dev/null)
if [ "$TELEGRAM_TEST" = "ok" ]; then
    echo -e "${GREEN}✅ Telegram is configured and ready${NC}"
else
    echo -e "${RED}❌ Telegram configuration issue${NC}"
fi

echo ""
echo "🔍 Checking corporate actions API..."
CORP_ACTIONS=$(curl -s "$BASE_URL/api/corporate_actions" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('actions',{})))" 2>/dev/null)
if [ "$CORP_ACTIONS" -gt 0 ]; then
    echo -e "${GREEN}✅ Corporate actions API working ($CORP_ACTIONS symbols)${NC}"
else
    echo -e "${RED}❌ Corporate actions API issue${NC}"
fi

echo ""
echo "🔄 Checking price diagnostics..."
DIAG_RESP=$(curl -s "$BASE_URL/api/price/diagnostics" 2>/dev/null)
if [ ! -z "$DIAG_RESP" ]; then
    echo -e "${GREEN}✅ Price diagnostics endpoint responding${NC}"
    
    # Check for backoff fields
    HAS_BACKOFF=$(echo "$DIAG_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if 'backoff_active' in d else 'no')" 2>/dev/null)
    if [ "$HAS_BACKOFF" = "yes" ]; then
        echo -e "${GREEN}   ├─ backoff_active field present${NC}"
    else
        echo -e "${YELLOW}   ├─ backoff_active field missing${NC}"
    fi
else
    echo -e "${RED}❌ Price diagnostics endpoint issue${NC}"
fi

echo ""
echo "📁 Checking documentation files..."
for doc in MARKET_OPEN_ALERT_FIX.md PROMETHEUS_METRICS_DEBUG.md COCKPIT_REFACTORING_PLAN.md SPRINT_SUMMARY_PHASE4.md; do
    if [ -f "/workspaces/GHOST/$doc" ]; then
        echo -e "${GREEN}✅ $doc${NC}"
    else
        echo -e "${RED}❌ $doc missing${NC}"
    fi
done

echo ""
echo "🧪 Checking test files..."
if [ -f "/workspaces/GHOST/tests/test_provider_backoff.py" ]; then
    echo -e "${GREEN}✅ test_provider_backoff.py created${NC}"
    TEST_COUNT=$(grep -c "^async def test_" /workspaces/GHOST/tests/test_provider_backoff.py)
    echo "   ├─ $TEST_COUNT test functions"
else
    echo -e "${RED}❌ test_provider_backoff.py missing${NC}"
fi

echo ""
echo "=============================================="
echo "📋 Summary:"
echo "   - Market open alerts: See MARKET_OPEN_ALERT_FIX.md"
echo "   - Metrics debugging: See PROMETHEUS_METRICS_DEBUG.md"
echo "   - Cockpit refactoring: See COCKPIT_REFACTORING_PLAN.md"
echo "   - Sprint summary: See SPRINT_SUMMARY_PHASE4.md"
echo ""
echo "🚀 Next Steps:"
echo "   1. If scheduler disabled, restart server with new config"
echo "   2. Monitor logs on next market open (9:30-9:40 AM ET)"
echo "   3. Run: pytest tests/test_provider_backoff.py -v"
echo "   4. Consider implementing eager metrics initialization"
echo ""

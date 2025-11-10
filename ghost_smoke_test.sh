#!/bin/bash
# Ghost Protocol Daily Smoke Test
# Comprehensive system health verification with alerting

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🧪 Ghost Protocol Daily Smoke Test${NC}"
echo "========================================"

# Environment setup
export HOST_URL="${HOST_URL:-http://127.0.0.1:5000}"
export AUTH="Authorization: Bearer $GHOST_API_TOKEN"
TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
SNAPSHOT_DIR="snapshots/daily_${TIMESTAMP}"
mkdir -p "$SNAPSHOT_DIR"

if [ -z "$GHOST_API_TOKEN" ]; then
    echo -e "${RED}❌ GHOST_API_TOKEN not set${NC}"
    exit 1
fi

echo -e "${BLUE}📊 Testing All Endpoints${NC}"
echo "-------------------------"

# Function to test endpoint and save snapshot
test_endpoint() {
    local name="$1"
    local endpoint="$2"
    local expected_key="$3"
    
    echo -n "Testing $name... "
    
    response=$(curl -s -H "$AUTH" "$HOST_URL$endpoint" 2>/dev/null)
    echo "$response" > "$SNAPSHOT_DIR/snap_${name}.json"
    
    if echo "$response" | grep -q "\"$expected_key\""; then
        echo -e "${GREEN}✅${NC}"
        return 0
    else
        echo -e "${RED}❌${NC}"
        echo "Error response: $response" > "$SNAPSHOT_DIR/error_${name}.txt"
        return 1
    fi
}

# Test all critical endpoints
FAILED_TESTS=0

test_endpoint "diagnostics" "/diagnostics" "ok" || ((FAILED_TESTS++))
test_endpoint "portfolio" "/portfolio" "total_usd" || ((FAILED_TESTS++))
test_endpoint "stocks" "/stocks" "items" || ((FAILED_TESTS++))
test_endpoint "fusion" "/fusionai" "rows" || ((FAILED_TESTS++))
test_endpoint "ghost" "/ghostscore" "score" || ((FAILED_TESTS++))
test_endpoint "source_status" "/source/status" "sources" || ((FAILED_TESTS++))
test_endpoint "presales" "/presales" "items" || ((FAILED_TESTS++))

echo
echo -e "${BLUE}📊 System Health Analysis${NC}"
echo "-------------------------"

# Analyze results
PORTFOLIO_VALUE=$(grep -o '"total_usd":[^,}]*' "$SNAPSHOT_DIR/snap_portfolio.json" | cut -d: -f2 2>/dev/null || echo "0")
STOCKS_COUNT=$(grep -o '"symbol"' "$SNAPSHOT_DIR/snap_stocks.json" | wc -l 2>/dev/null || echo "0")
FUSION_ROWS=$(grep -o '"symbol"' "$SNAPSHOT_DIR/snap_fusion.json" | wc -l 2>/dev/null || echo "0")

echo "Portfolio Value: \$$PORTFOLIO_VALUE"
echo "Stock Symbols: $STOCKS_COUNT"
echo "Fusion Analysis: $FUSION_ROWS rows"
echo "Failed Tests: $FAILED_TESTS"

# Generate summary report
cat > "$SNAPSHOT_DIR/summary.json" << EOF
{
  "timestamp": "$(date -u --iso-8601=seconds)",
  "test_results": {
    "total_tests": 7,
    "failed_tests": $FAILED_TESTS,
    "success_rate": $(( (7 - FAILED_TESTS) * 100 / 7 ))
  },
  "metrics": {
    "portfolio_value": $PORTFOLIO_VALUE,
    "stocks_count": $STOCKS_COUNT,
    "fusion_rows": $FUSION_ROWS
  },
  "status": "$([ $FAILED_TESTS -eq 0 ] && echo "healthy" || echo "degraded")"
}
EOF

echo
echo -e "${BLUE}🚨 Alert Check${NC}"
echo "---------------"

# Check for significant issues and send alerts
if [ "$FAILED_TESTS" -gt 2 ]; then
    echo -e "${RED}🚨 CRITICAL: Multiple system failures detected${NC}"
    
    # Discord alert if configured
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        python3 -c "
import sys
sys.path.append('.')
from discord_alerts import send_discord_alert
send_discord_alert(
    '🚨 Critical System Alert',
    f'Daily smoke test failed: {$FAILED_TESTS}/7 tests failed',
    0xff0000,
    [{'name': 'Portfolio Value', 'value': f'\$${$PORTFOLIO_VALUE}', 'inline': True}]
)
"
        echo "Discord alert sent"
    fi
    
elif [ "$FAILED_TESTS" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Warning: Some tests failed${NC}"
    
    if [ -n "$DISCORD_WEBHOOK_URL" ]; then
        python3 -c "
import sys
sys.path.append('.')
from discord_alerts import send_discord_alert
send_discord_alert(
    '⚠️ System Warning',
    f'Daily smoke test issues: {$FAILED_TESTS}/7 tests failed',
    0xffaa00
)
"
        echo "Discord warning sent"
    fi
    
else
    echo -e "${GREEN}✅ All systems operational${NC}"
fi

# Portfolio change detection
PREVIOUS_PORTFOLIO=$(find snapshots -name "snap_portfolio.json" -mtime -1 | head -1)
if [ -n "$PREVIOUS_PORTFOLIO" ]; then
    PREV_VALUE=$(grep -o '"total_usd":[^,}]*' "$PREVIOUS_PORTFOLIO" | cut -d: -f2 2>/dev/null || echo "$PORTFOLIO_VALUE")
    CHANGE_PCT=$(python3 -c "
try:
    change = (($PORTFOLIO_VALUE - $PREV_VALUE) / $PREV_VALUE * 100) if $PREV_VALUE > 0 else 0
    print(f'{change:.1f}')
except: print('0.0')
")
    
    echo "Portfolio change: ${CHANGE_PCT}%"
    
    # Alert on significant portfolio changes (>5%)
    if python3 -c "exit(0 if abs($CHANGE_PCT) >= 5.0 else 1)" 2>/dev/null; then
        if [ -n "$DISCORD_WEBHOOK_URL" ]; then
            python3 -c "
import sys
sys.path.append('.')
from discord_alerts import alert_portfolio_change
alert_portfolio_change($PREV_VALUE, $PORTFOLIO_VALUE)
"
            echo "Portfolio change alert sent"
        fi
    fi
fi

echo
echo -e "${BLUE}📁 Artifacts Saved${NC}"
echo "------------------"
echo "Snapshots: $SNAPSHOT_DIR/"
echo "Summary: $SNAPSHOT_DIR/summary.json"

# Cleanup old snapshots (keep last 7 days)
find snapshots -type d -name "daily_*" -mtime +7 -exec rm -rf {} + 2>/dev/null || true

echo
if [ "$FAILED_TESTS" -eq 0 ]; then
    echo -e "${GREEN}🎯 Daily smoke test: PASSED${NC}"
    exit 0
else
    echo -e "${RED}🚨 Daily smoke test: FAILED ($FAILED_TESTS errors)${NC}"
    exit 1
fi
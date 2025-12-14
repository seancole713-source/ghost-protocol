#!/bin/bash
# Ghost Protocol - Railway Health Monitor
# Checks production deployment health and reports status
#
# Usage: ./railway_health_monitor.sh
#        ./railway_health_monitor.sh --verbose

set -e

RAILWAY_URL="${RAILWAY_URL:-https://ghost-protocol-production.up.railway.app}"
VERBOSE=${1:-""}

echo "🔍 Ghost Protocol Health Monitor"
echo "=================================="
echo "Target: $RAILWAY_URL"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_endpoint() {
    local endpoint=$1
    local name=$2
    local expected_status=${3:-200}
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$RAILWAY_URL$endpoint" --max-time 10)
    
    if [ "$response" -eq "$expected_status" ]; then
        echo -e "${GREEN}✓${NC} $name (HTTP $response)"
        return 0
    else
        echo -e "${RED}✗${NC} $name (HTTP $response, expected $expected_status)"
        return 1
    fi
}

check_json_endpoint() {
    local endpoint=$1
    local name=$2
    local check_field=$3
    
    response=$(curl -s "$RAILWAY_URL$endpoint" --max-time 10)
    
    if echo "$response" | jq -e "$check_field" > /dev/null 2>&1; then
        value=$(echo "$response" | jq -r "$check_field")
        echo -e "${GREEN}✓${NC} $name ($check_field = $value)"
        
        if [ "$VERBOSE" == "--verbose" ]; then
            echo "$response" | jq '.'
        fi
        return 0
    else
        echo -e "${RED}✗${NC} $name (field '$check_field' not found)"
        if [ "$VERBOSE" == "--verbose" ]; then
            echo "$response"
        fi
        return 1
    fi
}

failures=0

echo "📊 Core Endpoints"
echo "-----------------"
check_endpoint "/health" "Health Check" || ((failures++))
check_endpoint "/api/v3/cockpit/status" "Cockpit Status" || ((failures++))

echo ""
echo "🎯 Prediction System"
echo "--------------------"
check_json_endpoint "/api/v3/predictions/latest" "Latest Predictions" ".ok" || ((failures++))
check_json_endpoint "/api/v3/hunter/feed" "Hunter Feed" ".ok" || ((failures++))

echo ""
echo "📈 Data Endpoints"
echo "-----------------"
check_json_endpoint "/api/v3/accuracy/summary" "Accuracy Summary" ".daily_accuracy_pct" || ((failures++))
check_json_endpoint "/api/v3/watchlist/user" "User Watchlist" ".ok" || ((failures++))
check_json_endpoint "/api/v3/goals/snapshot" "Goals Snapshot" ".ok" || ((failures++))

echo ""
echo "🔧 System Metrics"
echo "-----------------"
check_json_endpoint "/api/v3/health/metrics" "Health Metrics" ".ok" || ((failures++))

echo ""
echo "=================================="
if [ $failures -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "🚀 Ghost Protocol is healthy and operational"
    exit 0
else
    echo -e "${RED}❌ $failures check(s) failed${NC}"
    echo ""
    echo "⚠️  Some endpoints are not responding correctly"
    echo "Check Railway logs: railway logs"
    exit 1
fi

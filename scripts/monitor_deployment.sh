#!/bin/bash
# Monitor Railway deployment status by checking endpoints

BASE_URL="https://ghost-protocol-production.up.railway.app"
GREEN="\033[0;32m"
RED="\033[0;31m"
YELLOW="\033[0;33m"
NC="\033[0m" # No Color

echo "=========================================="
echo "🔍 Monitoring Railway Deployment Status"
echo "=========================================="
echo ""

check_endpoint() {
    local endpoint=$1
    local name=$2
    local expected_pattern=$3
    
    echo -n "Testing $name... "
    response=$(curl -sS -w "\n%{http_code}" "$BASE_URL$endpoint" 2>&1)
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [ "$http_code" == "200" ]; then
        if echo "$body" | grep -q "$expected_pattern"; then
            echo -e "${GREEN}✅ PASS${NC} (200, contains '$expected_pattern')"
            return 0
        else
            echo -e "${YELLOW}⚠️  WARN${NC} (200, but missing '$expected_pattern')"
            return 1
        fi
    elif [ "$http_code" == "404" ]; then
        echo -e "${RED}❌ 404 Not Found${NC}"
        return 1
    elif [ "$http_code" == "401" ]; then
        echo -e "${RED}❌ 401 Unauthorized${NC}"
        return 1
    else
        echo -e "${RED}❌ $http_code${NC}"
        return 1
    fi
}

# Wait for deployment to complete
echo "⏳ Waiting 30 seconds for Railway to start deployment..."
sleep 30

attempts=0
max_attempts=10

while [ $attempts -lt $max_attempts ]; do
    attempts=$((attempts + 1))
    echo ""
    echo "📊 Attempt $attempts/$max_attempts"
    echo "----------------------------------------"
    
    # Test all three endpoints
    check1=0
    check2=0
    check3=0
    
    check_endpoint "/api/v3/watchlist/enriched" "Watchlist Enriched" '"ok": true' && check1=1
    check_endpoint "/api/v3/watchlist/user" "Watchlist User" '"ok": true' && check2=1
    check_endpoint "/api/recent_alerts?limit=3" "Recent Alerts" '"ok":' && check3=1
    
    total=$((check1 + check2 + check3))
    
    echo ""
    echo "Status: $total/3 endpoints working"
    
    if [ $total -eq 3 ]; then
        echo ""
        echo "=========================================="
        echo -e "${GREEN}🎉 ALL ENDPOINTS OPERATIONAL!${NC}"
        echo "=========================================="
        exit 0
    fi
    
    if [ $attempts -lt $max_attempts ]; then
        echo "⏳ Waiting 20 seconds before next check..."
        sleep 20
    fi
done

echo ""
echo "=========================================="
echo -e "${RED}❌ Deployment still incomplete after $max_attempts attempts${NC}"
echo "=========================================="
echo ""
echo "Manual verification needed:"
echo "  1. Check Railway dashboard for deployment status"
echo "  2. Review deployment logs for errors"
echo "  3. Ensure healthcheck is passing"
exit 1

#!/bin/bash
# Ghost Protocol Trading Advisory System - Operational Runbook
# Ensures system runs in advisory-only mode with real portfolio tracking

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Ghost Protocol Advisory System - Operations Check${NC}"
echo "========================================================="

# Set environment variables
export HOST_URL="${HOST_URL:-http://127.0.0.1:5000}"
export AUTH="Authorization: Bearer $GHOST_API_TOKEN"

if [ -z "$GHOST_API_TOKEN" ]; then
    echo -e "${RED}❌ GHOST_API_TOKEN not set${NC}"
    exit 1
fi

echo -e "${BLUE}📊 System State (Quick Sweep)${NC}"
echo "-----------------------------"

# Check system health - if jq is available use it, otherwise parse manually
if command -v jq >/dev/null 2>&1; then
    # With jq (preferred)
    curl -s -H "$AUTH" "$HOST_URL/diagnostics" | jq '{ok,version,auth:.checks.auth_enforced,trader_ready:.checks.trader_ready}'
    
    echo -e "\n${BLUE}📈 Data Sources${NC}"
    curl -s -H "$AUTH" "$HOST_URL/source/status" | jq '.sources'
    
    echo -e "\n${BLUE}💰 Portfolio${NC}"
    curl -s -H "$AUTH" "$HOST_URL/portfolio" | jq '{total:.total_usd,holdings:(.holdings|length),errors}'
    
    echo -e "\n${BLUE}📊 Stocks Sample${NC}"
    curl -s -H "$AUTH" "$HOST_URL/stocks" | jq '.items[0]' 2>/dev/null || echo "No stock data"
else
    # Manual parsing (fallback)
    echo -e "${YELLOW}Note: Install jq for better formatting${NC}"
    
    DIAG=$(curl -s -H "$AUTH" "$HOST_URL/diagnostics" 2>/dev/null)
    echo "$DIAG" | grep -o '"version":"[^"]*"' | sed 's/"version":"/Version: /' | sed 's/"//'
    echo "$DIAG" | grep -o '"ok":[^,}]*' | sed 's/"ok":/Status: /'
    
    echo -e "\n${BLUE}💰 Portfolio${NC}"
    PORTFOLIO=$(curl -s -H "$AUTH" "$HOST_URL/portfolio" 2>/dev/null)
    TOTAL=$(echo "$PORTFOLIO" | grep -o '"total_usd":[^,}]*' | cut -d: -f2)
    HOLDINGS_COUNT=$(echo "$PORTFOLIO" | grep -o '"holdings":\[[^]]*\]' | grep -o '{' | wc -l)
    echo "Total USD: \$$TOTAL"
    echo "Holdings: $HOLDINGS_COUNT tokens"
fi

echo
echo -e "${BLUE}🛡️  Safety Guardrails Check${NC}"
echo "---------------------------"

# Check critical safety settings
if [ -z "$EXECUTE_TRADES" ] || [ "$EXECUTE_TRADES" = "0" ]; then
    echo -e "${GREEN}✅ EXECUTE_TRADES: UNSET/DISABLED (Advisory Mode Active)${NC}"
else
    echo -e "${RED}⚠️  WARNING: EXECUTE_TRADES=$EXECUTE_TRADES (Live Trading!)${NC}"
fi

if [ -n "$SAFETY_MAX_USD" ]; then
    echo -e "${GREEN}✅ SAFETY_MAX_USD: \$$SAFETY_MAX_USD (failsafe active)${NC}"
else
    echo -e "${YELLOW}⚠️  SAFETY_MAX_USD: Not set${NC}"
fi

if [ -z "$COVALENT_KEY" ]; then
    echo -e "${GREEN}✅ COVALENT_KEY: Absent (Ethplorer primary)${NC}"
else
    echo -e "${YELLOW}⚠️  COVALENT_KEY: Present (may cause 402 errors)${NC}"
fi

echo
echo -e "${BLUE}⚡ Rate Limiting Config${NC}"
echo "----------------------"
echo "STOCKS_QPM: ${STOCKS_QPM:-unset} (target: 5)"
echo "STOCKS_QPD: ${STOCKS_QPD:-unset} (target: 25)"
echo "STOCKS_REFRESH_SEC: ${STOCKS_REFRESH_SEC:-unset} (target: 10800 = 3h cache)"

echo
echo -e "${BLUE}🔧 Troubleshooting${NC}"
echo "------------------"
echo "If portfolio shows \$0:"
echo "  1. Confirm ETHPLORER_KEY is set"
echo "  2. Confirm COVALENT_KEY is removed"
echo "  3. Run: curl -s -H \"\$AUTH\" \"\$HOST_URL/refresh/all\" >/dev/null"
echo
echo "If stocks show \$0:"
echo "  1. Likely AlphaVantage daily quota hit (25/day)"
echo "  2. Check Finnhub/Polygon fallbacks if configured"
echo "  3. Prices resume after quota reset"
echo "  4. 3-hour cache keeps API usage sustainable"

echo
echo -e "${BLUE}📊 Midnight Stocks Check${NC}"
echo "-----------------------"
echo "To enable automated midnight stock price verification:"
echo "1. Set FINNHUB_KEY and/or POLYGON_KEY for fallback APIs"
echo "2. Run: ./ghost_ops.sh --midnight-check"
echo "3. Add to cron: 0 0 * * * cd /path/to/project && ./ghost_ops.sh --midnight-check"

echo
echo -e "${BLUE}📋 Operations Summary${NC}"
echo "--------------------"
if [ -z "$EXECUTE_TRADES" ] || [ "$EXECUTE_TRADES" = "0" ]; then
    echo -e "${GREEN}🎯 ADVISORY MODE: Real wallet balances live, quotes available, execution locked out${NC}"
else
    echo -e "${RED}🚨 LIVE TRADING MODE: Execution enabled!${NC}"
fi

# Refresh option
if [ "$1" = "--refresh" ]; then
    echo
    echo -e "${BLUE}🔄 Refreshing all data sources...${NC}"
    curl -s -H "$AUTH" "$HOST_URL/refresh/all" >/dev/null
    echo -e "${GREEN}✅ Refresh complete - data caches updated${NC}"
fi

# Midnight stocks check option
if [ "$1" = "--midnight-check" ]; then
    echo
    echo -e "${BLUE}🌙 Midnight Stocks Check${NC}"
    echo "Refreshing stock data and checking for zero prices..."
    
    # Refresh stocks
    curl -s -H "$AUTH" "$HOST_URL/refresh/all" >/dev/null
    
    # Save snapshot
    mkdir -p snapshots
    TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
    curl -s -H "$AUTH" "$HOST_URL/stocks" > "snapshots/snap_stocks_${TIMESTAMP}.json"
    
    # Check for zero prices and alert
    ZERO_COUNT=$(curl -s -H "$AUTH" "$HOST_URL/stocks" | grep -o '"price":0.0' | wc -l)
    if [ "$ZERO_COUNT" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Warning: $ZERO_COUNT stocks showing \$0.00 prices${NC}"
        # Discord alert if webhook configured
        if [ -n "$DISCORD_WEBHOOK_URL" ]; then
            python3 -c "
import sys
sys.path.append('.')
from discord_alerts import alert_api_quota_exhausted
alert_api_quota_exhausted('Stock APIs', 'Check fallback providers')
"
        fi
    else
        echo -e "${GREEN}✅ All stocks showing valid prices${NC}"
    fi
    exit 0
fi

# Keep-alive option
if [ "$1" = "--keepalive" ]; then
    echo
    echo -e "${BLUE}🔄 Starting keep-alive ping (15min intervals)${NC}"
    echo "Press Ctrl+C to stop..."
    while true; do
        curl -s -H "$AUTH" "$HOST_URL/refresh/all" >/dev/null
        echo -e "${GREEN}$(date): Cache refresh complete${NC}"
        sleep 900  # 15 minutes
    done
fi

echo
echo "🚀 Ghost Protocol Advisory System ready for operations!"
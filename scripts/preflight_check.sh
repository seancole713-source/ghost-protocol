#!/bin/bash
# Ghost Pre-Flight Checklist for Market Open
# Run this before market open to verify all systems are ready

set -e

GHOST_URL=${GHOST_URL:-http://localhost:5000}
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "============================================"
echo "   GHOST PRE-FLIGHT CHECKLIST"
echo "   $(date)"
echo "============================================"
echo ""

# Function to check and print result
check_item() {
    local test_name=$1
    local test_command=$2
    local expected=$3
    
    echo -n "Checking $test_name... "
    result=$(eval "$test_command" 2>/dev/null || echo "FAILED")
    
    if [[ "$result" == *"$expected"* ]] || [[ "$expected" == "ANY" && "$result" != "FAILED" ]]; then
        echo -e "${GREEN}✓ PASS${NC}"
        return 0
    else
        echo -e "${RED}✗ FAIL${NC}"
        echo "  Expected: $expected"
        echo "  Got: $result"
        return 1
    fi
}

echo "1. SERVER HEALTH"
echo "─────────────────────────────────────────"
check_item "Server responds" "curl -s -o /dev/null -w '%{http_code}' $GHOST_URL/health" "200"
check_item "Health endpoint OK" "curl -s $GHOST_URL/health | jq -r '.ok'" "true"
echo ""

echo "2. API KEYS & PROVIDERS"
echo "─────────────────────────────────────────"
if [ -n "$POLYGON_API_KEY" ]; then
    echo -e "Polygon API Key: ${GREEN}✓ Set${NC}"
else
    echo -e "Polygon API Key: ${YELLOW}⚠ Not Set${NC}"
fi

if [ -n "$ALPHAVANTAGE_API_KEY" ]; then
    echo -e "AlphaVantage API Key: ${GREEN}✓ Set${NC}"
else
    echo -e "AlphaVantage API Key: ${YELLOW}⚠ Not Set${NC}"
fi

if [ -n "$GHOST_API_TOKEN" ]; then
    echo -e "Ghost API Token: ${GREEN}✓ Set${NC}"
else
    echo -e "Ghost API Token: ${RED}✗ NOT SET${NC}"
fi
echo ""

echo "3. POSITION & PORTFOLIO"
echo "─────────────────────────────────────────"
position_qty=$(curl -s $GHOST_URL/api/cockpit | jq -r '.portfolio.qty')
position_avg=$(curl -s $GHOST_URL/api/cockpit | jq -r '.portfolio.avg_cost')
position_symbol=$(curl -s $GHOST_URL/api/cockpit | jq -r '.portfolio.symbol')

if [ "$position_qty" != "0" ] && [ "$position_qty" != "0.0" ] && [ "$position_qty" != "null" ]; then
    echo -e "Position Loaded: ${GREEN}✓ YES${NC}"
    echo "  Symbol: $position_symbol"
    echo "  Quantity: $position_qty"
    echo "  Avg Cost: \$$position_avg"
else
    echo -e "Position Loaded: ${RED}✗ NO POSITION${NC}"
    echo "  ⚠ You need to import a position!"
fi
echo ""

echo "4. PRICE FETCHING"
echo "─────────────────────────────────────────"
current_price=$(curl -s $GHOST_URL/api/price/WOLF | jq -r '.price')
price_provider=$(curl -s $GHOST_URL/api/price/WOLF | jq -r '.provider')
price_change=$(curl -s $GHOST_URL/api/price/WOLF | jq -r '.change_pct')

echo "Current Price: \$$current_price"
echo "Provider: $price_provider"
echo "Change: ${price_change}%"

if [ "$price_provider" == "prev-close" ]; then
    echo -e "Provider Status: ${YELLOW}⚠ Using prev-close (expected after hours)${NC}"
elif [ "$price_provider" == "unavailable" ]; then
    echo -e "Provider Status: ${RED}✗ NO PROVIDER${NC}"
else
    echo -e "Provider Status: ${GREEN}✓ LIVE DATA${NC}"
fi
echo ""

echo "5. RUNTIME CONFIGURATION"
echo "─────────────────────────────────────────"
config=$(curl -s $GHOST_URL/api/runtime/config)
echo "Price TTL (market open): $(echo $config | jq -r '.price_ttl_open_s')s"
echo "Price TTL (market closed): $(echo $config | jq -r '.price_ttl_s')s"
echo "Yahoo First: $(echo $config | jq -r '.yahoo_first')"
echo "Reuters Feeds: $(echo $config | jq -r '.reuters_feeds_on')"
echo "Diag Collapse Dupes: $(echo $config | jq -r '.diag_collapse_dupes')"
echo ""

echo "6. FLAGS & ANOMALIES"
echo "─────────────────────────────────────────"
flags=$(curl -s $GHOST_URL/api/cockpit | jq -r '.flags')
degraded=$(echo $flags | jq -r '.degraded')
price_anomaly=$(echo $flags | jq -r '.price_anomaly')
market_open=$(echo $flags | jq -r '.market_open')

if [ "$degraded" == "false" ]; then
    echo -e "Degraded: ${GREEN}✓ NO${NC}"
else
    echo -e "Degraded: ${YELLOW}⚠ YES${NC}"
fi

if [ "$price_anomaly" == "false" ]; then
    echo -e "Price Anomaly: ${GREEN}✓ NO${NC}"
else
    echo -e "Price Anomaly: ${YELLOW}⚠ YES${NC}"
fi

if [ "$market_open" == "true" ]; then
    echo -e "Market Open: ${GREEN}✓ YES${NC}"
else
    echo -e "Market Open: ${YELLOW}⚠ CLOSED (expected after hours)${NC}"
fi
echo ""

echo "7. FORECAST STATUS"
echo "─────────────────────────────────────────"
forecast_enabled=$(curl -s $GHOST_URL/api/cockpit | jq -r '.forecast_summary.enabled')
forecast_conf=$(curl -s $GHOST_URL/api/cockpit | jq -r '.forecast_summary.confidence')
forecast_horizon=$(curl -s $GHOST_URL/api/cockpit | jq -r '.forecast_summary.horizon_h')

if [ "$forecast_enabled" == "true" ]; then
    echo -e "Forecast Enabled: ${GREEN}✓ YES${NC}"
    echo "  Confidence: ${forecast_conf}%"
    echo "  Horizon: ${forecast_horizon}h"
else
    echo -e "Forecast Enabled: ${YELLOW}⚠ PAUSED${NC}"
fi
echo ""

echo "8. PROVIDER CIRCUIT BREAKERS"
echo "─────────────────────────────────────────"
providers=$(curl -s $GHOST_URL/diagnostics/summary | jq -r '.providers')
for provider in alphavantage polygon yahoo yfinance; do
    state=$(echo $providers | jq -r ".$provider.state")
    if [ "$state" == "closed" ]; then
        echo -e "$provider: ${GREEN}✓ READY${NC}"
    else
        backoff=$(echo $providers | jq -r ".$provider.backoff_factor")
        echo -e "$provider: ${YELLOW}⚠ OPEN (backoff: $backoff)${NC}"
    fi
done
echo ""

echo "9. NAV & P&L CALCULATION"
echo "─────────────────────────────────────────"
kpis=$(curl -s $GHOST_URL/api/cockpit | jq -r '.kpis')
nav=$(echo $kpis | jq -r '.nav')
pnl_abs=$(echo $kpis | jq -r '.pnl_abs')
pnl_pct=$(echo $kpis | jq -r '.pnl_pct')

echo "NAV: \$$nav"
echo "P&L: \$$pnl_abs ($pnl_pct%)"

if [ "$nav" != "0" ] && [ "$nav" != "0.0" ]; then
    echo -e "NAV Status: ${GREEN}✓ CALCULATED${NC}"
else
    echo -e "NAV Status: ${YELLOW}⚠ ZERO (no position?)${NC}"
fi
echo ""

echo "10. LOGS & ERRORS"
echo "─────────────────────────────────────────"
if [ -f /tmp/ghost_server.log ]; then
    error_count=$(grep -i "error" /tmp/ghost_server.log | tail -20 | wc -l)
    if [ "$error_count" -lt 5 ]; then
        echo -e "Recent Errors: ${GREEN}✓ FEW ($error_count)${NC}"
    else
        echo -e "Recent Errors: ${YELLOW}⚠ MANY ($error_count)${NC}"
        echo "  Review /tmp/ghost_server.log"
    fi
else
    echo -e "Server Log: ${YELLOW}⚠ NOT FOUND${NC}"
fi
echo ""

echo "============================================"
echo "   PRE-FLIGHT CHECK COMPLETE"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. If position is missing, run:"
echo "   LIVE_QTY=\$(railway variables get WOLF_QTY)"
echo "   LIVE_AVG=\$(railway variables get WOLF_AVG_COST)"
echo "   curl -X POST $GHOST_URL/api/positions/import \\\
    -H 'Authorization: Bearer \$GHOST_API_TOKEN' \\\
    -H 'Content-Type: application/json' \\\
    --data '{\"reset\":true,\"set_focus\":true,\"positions\":[{\"symbol\":\"WOLF\",\"qty\":'\"\$LIVE_QTY\"',\"avg_cost\":'\"\$LIVE_AVG\"'}]}'"
echo ""
echo "2. Monitor at market open (9:30 AM ET) to verify live prices"
echo ""
echo "3. Check UI at: $GHOST_URL/"
echo ""

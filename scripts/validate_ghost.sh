#!/usr/bin/env bash
# Ghost Production Validation Script
# Verifies all critical endpoints and data integrity

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

GHOST_URL="${GHOST_URL:-http://localhost:5000}"
FAILURES=0

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}   Ghost Production Validation Report${NC}"
echo -e "${BLUE}   $(date)${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Helper functions
check_endpoint() {
    local name="$1"
    local endpoint="$2"
    local timeout="${3:-5}"
    
    echo -n "[$name] Checking $endpoint... "
    
    if response=$(curl -fsS -m "$timeout" "$GHOST_URL$endpoint" 2>&1); then
        # Validate JSON
        if echo "$response" | jq empty 2>/dev/null; then
            echo -e "${GREEN}✓ OK${NC}"
            echo "$response"
        else
            echo -e "${YELLOW}⚠ Response not valid JSON${NC}"
            echo "$response" | head -n 5
            FAILURES=$((FAILURES + 1))
            return 1
        fi
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "Error: $response" | head -n 3
        FAILURES=$((FAILURES + 1))
        return 1
    fi
}

check_field() {
    local name="$1"
    local data="$2"
    local field="$3"
    local expected_type="$4"  # "not_null", "positive", "array", "string", etc.
    
    value=$(echo "$data" | jq -r "$field")
    
    case "$expected_type" in
        "not_null")
            if [ "$value" = "null" ]; then
                echo -e "${RED}✗${NC} $name: $field is null"
                FAILURES=$((FAILURES + 1))
                return 1
            else
                echo -e "${GREEN}✓${NC} $name: $field = $value"
            fi
            ;;
        "positive")
            if [ "$value" != "null" ] && [ "$(echo "$value > 0" | bc 2>/dev/null)" = "1" ]; then
                echo -e "${GREEN}✓${NC} $name: $field = $value"
            else
                echo -e "${YELLOW}⚠${NC} $name: $field = $value (expected > 0)"
            fi
            ;;
        "array")
            length=$(echo "$data" | jq "$field | length")
            if [ "$length" -gt 0 ]; then
                echo -e "${GREEN}✓${NC} $name: $field has $length items"
            else
                echo -e "${YELLOW}⚠${NC} $name: $field is empty array"
            fi
            ;;
        "string")
            if [ "$value" != "null" ] && [ -n "$value" ]; then
                echo -e "${GREEN}✓${NC} $name: $field = \"$value\""
            else
                echo -e "${RED}✗${NC} $name: $field is null or empty"
                FAILURES=$((FAILURES + 1))
                return 1
            fi
            ;;
    esac
}

echo -e "${BLUE}━━━ Health Check ━━━${NC}"
health=$(check_endpoint "Health" "/health" 2)
if [ $? -eq 0 ]; then
    check_field "Health" "$health" ".ok" "not_null"
    check_field "Health" "$health" ".ts" "positive"
fi
echo ""

echo -e "${BLUE}━━━ Positions API ━━━${NC}"
positions=$(check_endpoint "Positions" "/api/positions" 5)
if [ $? -eq 0 ]; then
    check_field "Positions" "$positions" ".positions" "array"
    qty=$(echo "$positions" | jq -r '.positions[0].qty // 0')
    symbol=$(echo "$positions" | jq -r '.positions[0].symbol // "NONE"')
    echo -e "${GREEN}✓${NC} First position: $symbol with $qty shares"
fi
echo ""

echo -e "${BLUE}━━━ Cockpit Snapshot ━━━${NC}"
cockpit=$(check_endpoint "Cockpit" "/api/cockpit" 15)
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}--- Prices ---${NC}"
    check_field "Prices" "$cockpit" ".prices.provider" "string"
    check_field "Prices" "$cockpit" ".prices.price" "not_null"
    
    # Check if we have latency/timestamp diagnostics
    latency=$(echo "$cockpit" | jq -r '.prices.latency_ms // "N/A"')
    if [ "$latency" != "N/A" ]; then
        echo -e "${GREEN}✓${NC} Price latency: ${latency}ms"
    fi
    
    echo ""
    echo -e "${YELLOW}--- Portfolio ---${NC}"
    check_field "Portfolio" "$cockpit" ".portfolio.symbol" "string"
    check_field "Portfolio" "$cockpit" ".portfolio.qty" "not_null"
    check_field "Portfolio" "$cockpit" ".portfolio.avg_cost" "not_null"
    check_field "Portfolio" "$cockpit" ".portfolio.rows" "array"
    
    echo ""
    echo -e "${YELLOW}--- KPIs ---${NC}"
    check_field "KPIs" "$cockpit" ".kpis.nav" "not_null"
    check_field "KPIs" "$cockpit" ".kpis.cash" "not_null"
    check_field "KPIs" "$cockpit" ".kpis.pnl_abs" "not_null"
    
    echo ""
    echo -e "${YELLOW}--- Forecast ---${NC}"
    has_forecast=$(echo "$cockpit" | jq -r '.forecast != null')
    if [ "$has_forecast" = "true" ]; then
        check_field "Forecast" "$cockpit" ".forecast_summary.enabled" "not_null"
        check_field "Forecast" "$cockpit" ".forecast_summary.horizon_h" "not_null"
        check_field "Forecast" "$cockpit" ".forecast_summary.confidence" "not_null"
        
        points=$(echo "$cockpit" | jq '.forecast.points | length // 0')
        if [ "$points" -gt 0 ]; then
            echo -e "${GREEN}✓${NC} Forecast has $points time-series points"
        else
            echo -e "${YELLOW}⚠${NC} Forecast has no points"
        fi
    else
        echo -e "${RED}✗${NC} Forecast is null"
        FAILURES=$((FAILURES + 1))
    fi
    
    echo ""
    echo -e "${YELLOW}--- Accuracy Metrics ---${NC}"
    has_metrics=$(echo "$cockpit" | jq -r '.metrics != null')
    if [ "$has_metrics" = "true" ]; then
        check_field "Metrics" "$cockpit" ".metrics.map" "not_null"
        check_field "Metrics" "$cockpit" ".metrics.rmse" "not_null"
        check_field "Metrics" "$cockpit" ".metrics.bias" "not_null"
    else
        echo -e "${YELLOW}⚠${NC} Metrics not yet available (need historical data)"
    fi
    
    echo ""
    echo -e "${YELLOW}--- Flags ---${NC}"
    degraded=$(echo "$cockpit" | jq -r '.flags.degraded')
    market_open=$(echo "$cockpit" | jq -r '.flags.market_open')
    using_prev=$(echo "$cockpit" | jq -r '.flags.using_prev_close')
    
    echo -e "  Degraded: $degraded"
    echo -e "  Market Open: $market_open"
    echo -e "  Using Prev Close: $using_prev"
fi
echo ""

echo -e "${BLUE}━━━ Diagnostics ━━━${NC}"
diag=$(check_endpoint "Diagnostics" "/diagnostics/summary" 5)
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}--- Price Diagnostics ---${NC}"
    provider=$(echo "$diag" | jq -r '.price_diag.last_fetch_provider // "N/A"')
    latency=$(echo "$diag" | jq -r '.price_diag.last_fetch_latency_ms // "N/A"')
    quorum=$(echo "$diag" | jq -r '.price_diag.quorum_ok // false')
    
    echo -e "  Last Provider: $provider"
    echo -e "  Latency: ${latency}ms"
    echo -e "  Quorum OK: $quorum"
    
    echo ""
    echo -e "${YELLOW}--- Recent Events ---${NC}"
    events=$(echo "$diag" | jq -r '.events[-5:] | .[] | "\(.type): \(.message)"' 2>/dev/null || echo "N/A")
    echo "$events" | while IFS= read -r line; do
        echo -e "  • $line"
    done
fi
echo ""

echo -e "${BLUE}━━━ Events Stream (SSE) ━━━${NC}"
echo -n "Checking SSE stream... "
if timeout 3 curl -fsS --no-buffer "$GHOST_URL/events" 2>&1 | head -n 5 > /tmp/ghost_events.txt; then
    event_count=$(grep -c "^data:" /tmp/ghost_events.txt || echo 0)
    if [ "$event_count" -gt 0 ]; then
        echo -e "${GREEN}✓ OK${NC} ($event_count events received)"
        echo -e "${YELLOW}Sample events:${NC}"
        head -n 3 /tmp/ghost_events.txt | sed 's/^/  /'
    else
        echo -e "${YELLOW}⚠ No events received${NC}"
    fi
else
    echo -e "${RED}✗ FAILED${NC}"
    FAILURES=$((FAILURES + 1))
fi
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
if [ $FAILURES -eq 0 ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "${GREEN}  Ghost is production-ready.${NC}"
    exit 0
else
    echo -e "${RED}✗ $FAILURES check(s) failed${NC}"
    echo -e "${RED}  Review errors above and fix issues.${NC}"
    exit 1
fi

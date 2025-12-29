#!/bin/bash
#
# 📊 GHOST ORACLE - DAILY MONITORING (Bash Version)
# ==================================================
# Quick daily check script using curl + jq
#
# Usage:
#   ./ghost_monitor.sh              # Full report
#   ./ghost_monitor.sh --quick      # Quick accuracy only
#
# Requirements: curl, jq
#

BASE_URL="${GHOST_URL:-https://ghost-protocol-production.up.railway.app}"

echo ""
echo "============================================================"
echo "🔍 GHOST ORACLE - DAILY MONITORING"
echo "⏰ $(date -u '+%Y-%m-%d %H:%M:%S') UTC"
echo "🌐 $BASE_URL"
echo "============================================================"

# Quick mode check
if [ "$1" == "--quick" ]; then
    echo ""
    echo "📊 QUICK ACCURACY CHECK"
    echo "------------------------------------------------------------"
    
    AUDIT=$(curl -s "$BASE_URL/debug/db-audit" 2>/dev/null)
    
    if [ -n "$AUDIT" ]; then
        WINS=$(echo "$AUDIT" | jq -r '.outcomes_stats.wins // 0')
        LOSSES=$(echo "$AUDIT" | jq -r '.outcomes_stats.losses // 0')
        TOTAL=$((WINS + LOSSES))
        
        if [ "$TOTAL" -gt 0 ]; then
            ACCURACY=$(echo "scale=2; $WINS * 100 / $TOTAL" | bc)
            echo "   📈 Accuracy: ${ACCURACY}%"
            echo "   ✅ Wins: $WINS"
            echo "   ❌ Losses: $LOSSES"
            echo "   📊 Total: $TOTAL"
        else
            echo "   ⚠️ No completed outcomes yet"
        fi
    else
        echo "   ❌ Could not fetch data"
    fi
    
    echo ""
    exit 0
fi

# Full report
echo ""
echo "🏥 SYSTEM HEALTH"
echo "------------------------------------------------------------"

HEALTH=$(curl -s "$BASE_URL/health" 2>/dev/null)
if [ -n "$HEALTH" ]; then
    STATUS=$(echo "$HEALTH" | jq -r '.status // "unknown"')
    if [ "$STATUS" == "healthy" ]; then
        echo "   ✅ Status: healthy"
    else
        echo "   ⚠️ Status: $STATUS"
    fi
else
    echo "   ❌ Could not fetch health status"
fi

echo ""
echo "📊 ACCURACY STATUS"
echo "------------------------------------------------------------"

AUDIT=$(curl -s "$BASE_URL/debug/db-audit" 2>/dev/null)

if [ -n "$AUDIT" ]; then
    TOTAL=$(echo "$AUDIT" | jq -r '.overview.total // 0')
    WINS=$(echo "$AUDIT" | jq -r '.outcomes_stats.wins // 0')
    LOSSES=$(echo "$AUDIT" | jq -r '.outcomes_stats.losses // 0')
    NO_DATA=$(echo "$AUDIT" | jq -r '.outcomes_stats.no_data // 0')
    COMPLETED=$((WINS + LOSSES))
    
    if [ "$COMPLETED" -gt 0 ]; then
        ACCURACY=$(echo "scale=2; $WINS * 100 / $COMPLETED" | bc)
        
        echo "   📈 Accuracy: ${ACCURACY}%"
        echo "   ✅ Wins: $WINS"
        echo "   ❌ Losses: $LOSSES"
        echo "   📊 Completed: $COMPLETED"
        echo "   📦 Total outcomes: $TOTAL"
        echo "   ⚠️ No data: $NO_DATA"
        
        # Progress bar (30 chars)
        FILLED=$(echo "$ACCURACY * 30 / 100" | bc 2>/dev/null || echo "0")
        FILLED=${FILLED:-0}
        EMPTY=$((30 - FILLED))
        
        BAR=""
        for ((i=0; i<FILLED; i++)); do BAR+="█"; done
        for ((i=0; i<EMPTY; i++)); do BAR+="░"; done
        
        echo ""
        echo "   Progress: [$BAR] ${ACCURACY}%"
        echo "   Target:   [█████████████████████░░░░░░░░░] 70.0%"
        
        # Status determination
        if (( $(echo "$ACCURACY >= 70" | bc -l) )); then
            echo ""
            echo "   🎉 STATUS: GOAL REACHED!"
        elif (( $(echo "$ACCURACY >= 65" | bc -l) )); then
            echo ""
            echo "   ⭐ STATUS: EXCELLENT"
        elif (( $(echo "$ACCURACY >= 60" | bc -l) )); then
            echo ""
            echo "   ✅ STATUS: GOOD"
        elif (( $(echo "$ACCURACY >= 55" | bc -l) )); then
            echo ""
            echo "   📈 STATUS: ACCEPTABLE"
        elif (( $(echo "$ACCURACY >= 50" | bc -l) )); then
            echo ""
            echo "   ⚠️ STATUS: BELOW TARGET"
        else
            echo ""
            echo "   🔴 STATUS: CRITICAL - Below 50%"
        fi
    else
        echo "   ⚠️ No completed outcomes yet"
    fi
    
    # Date range
    EARLIEST=$(echo "$AUDIT" | jq -r '.overview.earliest // "N/A"')
    LATEST=$(echo "$AUDIT" | jq -r '.overview.latest // "N/A"')
    echo ""
    echo "   📅 Date range: $EARLIEST to $LATEST"
    
    # Corrupt data check
    CORRUPT=$(echo "$AUDIT" | jq -r '.total_corrupt // 0')
    if [ "$CORRUPT" -gt 0 ]; then
        echo "   ⚠️ Corrupt records: $CORRUPT"
    else
        echo "   ✅ Data quality: Clean"
    fi
else
    echo "   ❌ Could not fetch audit data"
fi

echo ""
echo "🔄 INVERSE MODE STATUS"
echo "------------------------------------------------------------"

INVERSE=$(curl -s "$BASE_URL/debug/inverse-status" 2>/dev/null)

if [ -n "$INVERSE" ]; then
    MODE=$(echo "$INVERSE" | jq -r '.inverse_ghost_enabled // false')
    SKIP_COUNT=$(echo "$INVERSE" | jq -r '.inverse_skip_count // 0')
    
    if [ "$MODE" == "true" ]; then
        echo "   ✅ INVERSE mode: ENABLED"
    else
        echo "   ❌ INVERSE mode: DISABLED"
    fi
    
    echo "   ⏭️ Skip symbols (use RAW): $SKIP_COUNT"
    
    # Show sample symbol modes
    echo ""
    echo "   Sample symbol modes:"
    echo "$INVERSE" | jq -r '.symbol_modes | to_entries[:6][] | "      \(if .value | contains("INVERTED") then "🔄" else "⏭️" end) \(.key): \(.value | split("(")[0])"' 2>/dev/null
else
    echo "   ❌ Could not fetch inverse status"
fi

echo ""
echo "📅 RECENT OUTCOMES"
echo "------------------------------------------------------------"

OUTCOME_AUDIT=$(curl -s "$BASE_URL/debug/outcome-data-audit" 2>/dev/null)

if [ -n "$OUTCOME_AUDIT" ]; then
    EARLIEST=$(echo "$OUTCOME_AUDIT" | jq -r '.audit.date_range.earliest // "N/A"')
    NEWEST=$(echo "$OUTCOME_AUDIT" | jq -r '.audit.date_range.latest // "N/A"')
    
    echo "   📆 Date range: $EARLIEST to $NEWEST"
    
    # Show hit direction distribution
    echo ""
    echo "   Outcome distribution:"
    echo "$OUTCOME_AUDIT" | jq -r '.audit.hit_direction_distribution[] | "      \(if .hit_direction == 1 then "✅ Correct" elif .hit_direction == 0 then "❌ Wrong" else "❓ No data" end): \(.count)"' 2>/dev/null
else
    echo "   ❌ Could not fetch outcome data"
fi

echo ""
echo "🔮 TODAY'S PREDICTIONS"
echo "------------------------------------------------------------"

TOP10=$(curl -s "$BASE_URL/debug/top10-preview" 2>/dev/null)

if [ -n "$TOP10" ]; then
    PRED_COUNT=$(echo "$TOP10" | jq -r '.predictions | length // 0')
    
    if [ "$PRED_COUNT" -gt 0 ]; then
        echo "   📊 Total predictions: $PRED_COUNT"
        
        UP_COUNT=$(echo "$TOP10" | jq '[.predictions[] | select(.direction == "UP")] | length')
        DOWN_COUNT=$(echo "$TOP10" | jq '[.predictions[] | select(.direction == "DOWN")] | length')
        
        echo "   🟢 UP predictions: $UP_COUNT"
        echo "   🔴 DOWN predictions: $DOWN_COUNT"
        
        echo ""
        echo "   Top 5 predictions:"
        echo "$TOP10" | jq -r '.predictions[:5][] | "      \(if .direction == "UP" then "🟢" else "🔴" end) \(.symbol): \(.direction) (\(.confidence * 100 | floor)%)"' 2>/dev/null
    else
        echo "   ⚠️ No predictions available"
    fi
else
    echo "   ❌ Could not fetch predictions"
fi

echo ""
echo "============================================================"
echo "💡 RECOMMENDATIONS"
echo "============================================================"

if [ -n "$AUDIT" ] && [ "$COMPLETED" -gt 0 ]; then
    if (( $(echo "$ACCURACY < 50" | bc -l) )); then
        echo "   🔴 CRITICAL: Accuracy below 50%"
        echo "      → Note: Old data from before INVERSE (Dec 18-21)"
        echo "      → Wait for new post-INVERSE data to accumulate"
        echo "      → Check again Dec 30-31 for improvement"
    elif (( $(echo "$ACCURACY < 55" | bc -l) )); then
        echo "   ⚠️ Below target - monitor closely"
        echo "      → Review symbol exclusions"
        echo "      → Wait for more post-INVERSE data"
    elif (( $(echo "$ACCURACY < 60" | bc -l) )); then
        echo "   📈 Making progress - keep monitoring"
        echo "      → Consider additional symbol exclusions"
    else
        echo "   ✅ On track! Keep monitoring daily"
    fi
else
    echo "   ⏳ Waiting for data to accumulate"
    echo "      → Check again in 24-48 hours"
fi

echo ""
echo "============================================================"
echo "📊 END OF REPORT"
echo "============================================================"
echo ""

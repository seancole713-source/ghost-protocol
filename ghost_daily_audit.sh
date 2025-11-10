#!/bin/bash
#
# Ghost 24-Hour Auto-Audit Script
# ================================
# Self-checks memory, data, and Telegram functions daily
# Run this via cron: 0 9 * * * /path/to/ghost_daily_audit.sh
#

set -e

# Configuration
GHOST_URL="${GHOST_URL:-https://web-production-8e9a0.up.railway.app}"
API_TOKEN="${GHOST_API_TOKEN:-}"
TELEGRAM_BOT_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT_ID="${TELEGRAM_CHAT_ID:-}"
LOG_FILE="${LOG_FILE:-/tmp/ghost_audit_$(date +%Y%m%d).log}"

# Audit results
PASSED=0
FAILED=0
WARNINGS=0
AUDIT_ISSUES=()

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

log_pass() {
    echo -e "${GREEN}✅ PASS${NC} - $*" | tee -a "$LOG_FILE"
    PASSED=$((PASSED + 1))
}

log_fail() {
    echo -e "${RED}❌ FAIL${NC} - $*" | tee -a "$LOG_FILE"
    FAILED=$((FAILED + 1))
    AUDIT_ISSUES+=("$*")
}

log_warn() {
    echo -e "${YELLOW}⚠️  WARN${NC} - $*" | tee -a "$LOG_FILE"
    WARNINGS=$((WARNINGS + 1))
}

log_info() {
    echo -e "${BLUE}ℹ️  INFO${NC} - $*" | tee -a "$LOG_FILE"
}

# Check if jq is available
if ! command -v jq &> /dev/null; then
    log_fail "jq not installed - required for JSON parsing"
    exit 1
fi

# Start audit
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "🤖 Ghost Daily Auto-Audit - $(date +'%Y-%m-%d %H:%M:%S')"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log_info "Target: $GHOST_URL"
log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "📊 1. CORE HEALTH CHECK"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Basic health
response=$(curl -s "$GHOST_URL/health" || echo '{"ok":false}')
if echo "$response" | jq -e '.ok == true' > /dev/null 2>&1; then
    log_pass "Basic health endpoint responding"
else
    log_fail "Basic health endpoint failed or returned ok=false"
fi

# Detailed health
response=$(curl -s "$GHOST_URL/health/detailed" || echo '{}')
if echo "$response" | jq -e '.ok' > /dev/null 2>&1; then
    overall_ok=$(echo "$response" | jq -r '.ok')
    if [ "$overall_ok" = "true" ]; then
        log_pass "Detailed health check: all systems operational"
    else
        log_fail "Detailed health check reports issues"
        # Log specific issues
        issues=$(echo "$response" | jq -r '.issues[]' 2>/dev/null || echo "")
        if [ -n "$issues" ]; then
            while IFS= read -r issue; do
                log_warn "  → $issue"
            done <<< "$issues"
        fi
    fi
else
    log_fail "Detailed health endpoint not responding"
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "🧠 2. AI MEMORY AUDIT"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -z "$API_TOKEN" ]; then
    log_warn "Skipping AI memory checks (no API token)"
else
    # Check AI memory stats
    response=$(curl -s -H "Authorization: Bearer $API_TOKEN" "$GHOST_URL/ai/memory/stats" || echo '{}')
    
    if echo "$response" | jq -e '.count' > /dev/null 2>&1; then
        count=$(echo "$response" | jq -r '.count')
        last_ts=$(echo "$response" | jq -r '.last_ts')
        
        log_pass "AI Memory accessible: $count decisions stored"
        
        # Check if memory is growing (has recent activity)
        if [ "$last_ts" != "null" ] && [ -n "$last_ts" ]; then
            now=$(date +%s)
            age=$((now - last_ts))
            hours=$((age / 3600))
            
            if [ $hours -lt 48 ]; then
                log_pass "AI Memory recently active (last update: ${hours}h ago)"
            else
                log_warn "AI Memory stale (last update: ${hours}h ago)"
            fi
        fi
        
        # Check if memory count is reasonable
        if [ "$count" -gt 50000 ]; then
            log_pass "AI Memory well-populated: $count decisions"
        elif [ "$count" -gt 10000 ]; then
            log_info "AI Memory moderately populated: $count decisions"
        elif [ "$count" -gt 0 ]; then
            log_warn "AI Memory low: only $count decisions"
        else
            log_fail "AI Memory empty: 0 decisions"
        fi
    else
        log_fail "AI Memory stats endpoint not responding"
    fi
    
    # Check recent decisions
    response=$(curl -s -H "Authorization: Bearer $API_TOKEN" "$GHOST_URL/ai/memory/recent?limit=5" || echo '[]')
    
    if echo "$response" | jq -e '. | length' > /dev/null 2>&1; then
        recent_count=$(echo "$response" | jq '. | length')
        if [ "$recent_count" -gt 0 ]; then
            log_pass "Recent decisions retrievable ($recent_count records)"
        else
            log_warn "No recent decisions found"
        fi
    else
        log_fail "Recent decisions endpoint not responding"
    fi
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "💾 3. DATA PERSISTENCE AUDIT"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

response=$(curl -s "$GHOST_URL/health/detailed" || echo '{}')

# Check portfolio persistence
if echo "$response" | jq -e '.components.positions' > /dev/null 2>&1; then
    pos_ok=$(echo "$response" | jq -r '.components.positions.ok')
    
    if [ "$pos_ok" = "true" ]; then
        log_pass "Portfolio persistence layer operational"
        
        # Check position data
        wolf_qty=$(echo "$response" | jq -r '.components.positions.wolf_qty // 0')
        wolf_avg=$(echo "$response" | jq -r '.components.positions.wolf_avg // 0')
        
        if [ "$(echo "$wolf_qty > 0" | bc -l 2>/dev/null || echo 0)" = "1" ]; then
            log_info "Current position: $wolf_qty shares @ avg \$$wolf_avg"
        else
            log_info "Portfolio currently empty (0 shares)"
        fi
    else
        log_fail "Portfolio persistence layer error"
    fi
else
    log_fail "Portfolio persistence status unavailable"
fi

# Check AI memory database
if echo "$response" | jq -e '.components.ai_memory' > /dev/null 2>&1; then
    ai_ok=$(echo "$response" | jq -r '.components.ai_memory.ok')
    
    if [ "$ai_ok" = "true" ]; then
        records=$(echo "$response" | jq -r '.components.ai_memory.records // 0')
        log_pass "AI Memory database: $records records"
    else
        error=$(echo "$response" | jq -r '.components.ai_memory.error // "unknown"')
        log_fail "AI Memory database error: $error"
    fi
else
    log_fail "AI Memory database status unavailable"
fi

# Check price cache
if echo "$response" | jq -e '.components.cache' > /dev/null 2>&1; then
    cache_size=$(echo "$response" | jq -r '.components.cache.price_cache_size // 0')
    news_age=$(echo "$response" | jq -r '.components.cache.news_cache_age_s // 0')
    
    log_pass "Price cache: $cache_size symbols cached"
    
    # Check news cache freshness
    news_hours=$((news_age / 3600))
    if [ $news_hours -lt 6 ]; then
        log_pass "News cache fresh (${news_hours}h old)"
    elif [ $news_hours -lt 24 ]; then
        log_warn "News cache aging (${news_hours}h old)"
    else
        log_warn "News cache stale (${news_hours}h old)"
    fi
else
    log_fail "Cache status unavailable"
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "📈 4. PRICE PROVIDER AUDIT"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if echo "$response" | jq -e '.components.price_providers' > /dev/null 2>&1; then
    # Check current price
    price=$(echo "$response" | jq -r '.components.price_providers.current_price.price')
    provider=$(echo "$response" | jq -r '.components.price_providers.current_price.provider')
    price_ok=$(echo "$response" | jq -r '.components.price_providers.current_price.ok')
    
    if [ "$price_ok" = "true" ]; then
        log_pass "Current price available: \$$price (source: $provider)"
        
        # Warn if using fallback
        if [ "$provider" = "prev-close" ] || [ "$provider" = "cached" ]; then
            log_warn "Using fallback price source: $provider"
        fi
    else
        log_fail "Current price unavailable"
    fi
    
    # Check API keys
    av_key=$(echo "$response" | jq -r '.components.price_providers.api_keys.alphavantage')
    poly_key=$(echo "$response" | jq -r '.components.price_providers.api_keys.polygon')
    
    if [ "$av_key" = "true" ] && [ "$poly_key" = "true" ]; then
        log_pass "All premium API keys configured"
    elif [ "$av_key" = "true" ] || [ "$poly_key" = "true" ]; then
        log_warn "Some premium API keys missing"
    else
        log_warn "No premium API keys configured"
    fi
    
    # Check provider diagnostics
    fallback_reason=$(echo "$response" | jq -r '.components.price_providers.diagnostics.fallback_reason // "none"')
    if [ "$fallback_reason" != "none" ] && [ "$fallback_reason" != "null" ]; then
        log_warn "Price fallback active: $fallback_reason"
    fi
else
    log_fail "Price provider status unavailable"
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "📱 5. TELEGRAM BOT AUDIT"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    log_warn "Skipping Telegram checks (no bot token)"
else
    # Check bot status via Telegram API
    bot_response=$(curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getMe" || echo '{"ok":false}')
    
    if echo "$bot_response" | jq -e '.ok == true' > /dev/null 2>&1; then
        bot_name=$(echo "$bot_response" | jq -r '.result.username')
        bot_id=$(echo "$bot_response" | jq -r '.result.id')
        log_pass "Telegram bot active: @$bot_name (ID: $bot_id)"
    else
        log_fail "Telegram bot not responding or token invalid"
    fi
    
    # Check webhook configuration
    webhook_response=$(curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo" || echo '{"ok":false}')
    
    if echo "$webhook_response" | jq -e '.ok == true' > /dev/null 2>&1; then
        webhook_url=$(echo "$webhook_response" | jq -r '.result.url // ""')
        pending_count=$(echo "$webhook_response" | jq -r '.result.pending_update_count // 0')
        last_error=$(echo "$webhook_response" | jq -r '.result.last_error_message // ""')
        
        if [ -n "$webhook_url" ]; then
            log_pass "Webhook configured: $webhook_url"
            
            if [ "$pending_count" -gt 0 ]; then
                log_warn "Webhook has $pending_count pending updates"
            else
                log_pass "Webhook queue empty (no pending updates)"
            fi
            
            if [ -n "$last_error" ]; then
                log_warn "Last webhook error: $last_error"
            fi
        else
            log_fail "Webhook not configured"
        fi
    else
        log_fail "Cannot retrieve webhook info"
    fi
    
    # Check Ghost webhook endpoint
    webhook_status=$(curl -s -o /dev/null -w "%{http_code}" "$GHOST_URL/telegram/webhook")
    if [ "$webhook_status" = "200" ] || [ "$webhook_status" = "405" ]; then
        log_pass "Ghost webhook endpoint exists (HTTP $webhook_status)"
    else
        log_fail "Ghost webhook endpoint unavailable (HTTP $webhook_status)"
    fi
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "🔐 6. SECURITY AUDIT"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

response=$(curl -s "$GHOST_URL/api/secrets/health" || echo '{}')

if echo "$response" | jq -e '.present' > /dev/null 2>&1; then
    # Check each required secret
    ghost_token=$(echo "$response" | jq -r '.present.GHOST_API_TOKEN')
    av_key=$(echo "$response" | jq -r '.present.ALPHAVANTAGE_API_KEY')
    poly_key=$(echo "$response" | jq -r '.present.POLYGON_API_KEY')
    tg_token=$(echo "$response" | jq -r '.present.TELEGRAM_BOT_TOKEN')
    tg_chat=$(echo "$response" | jq -r '.present.TELEGRAM_CHAT_ID')
    
    total_secrets=0
    set_secrets=0
    
    for secret in "$ghost_token" "$av_key" "$poly_key" "$tg_token" "$tg_chat"; do
        total_secrets=$((total_secrets + 1))
        if [ "$secret" = "true" ]; then
            set_secrets=$((set_secrets + 1))
        fi
    done
    
    if [ $set_secrets -eq $total_secrets ]; then
        log_pass "All secrets configured ($set_secrets/$total_secrets)"
    else
        log_warn "Some secrets missing ($set_secrets/$total_secrets configured)"
    fi
    
    # List missing secrets
    missing=$(echo "$response" | jq -r '.missing[]' 2>/dev/null || echo "")
    if [ -n "$missing" ]; then
        while IFS= read -r secret; do
            log_warn "  → Missing: $secret"
        done <<< "$missing"
    fi
else
    log_fail "Secrets health endpoint not responding"
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "🎯 7. FUNCTIONALITY SPOT CHECKS"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check position endpoint
position_response=$(curl -s "$GHOST_URL/api/position" || echo '{}')
if echo "$position_response" | jq -e '.symbol' > /dev/null 2>&1; then
    log_pass "Position endpoint responding"
else
    log_fail "Position endpoint not responding"
fi

# Check version endpoint
version_response=$(curl -s "$GHOST_URL/api/version" || echo '{}')
if echo "$version_response" | jq -e '.version' > /dev/null 2>&1; then
    version=$(echo "$version_response" | jq -r '.version')
    log_pass "API version: $version"
else
    log_fail "Version endpoint not responding"
fi

# Check config endpoint
config_response=$(curl -s "$GHOST_URL/api/config" || echo '{}')
if echo "$config_response" | jq -e '.ticker' > /dev/null 2>&1; then
    ticker=$(echo "$config_response" | jq -r '.ticker')
    log_pass "Configuration accessible (tracking: $ticker)"
else
    log_fail "Config endpoint not responding"
fi

log ""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log "📊 AUDIT SUMMARY"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

TOTAL=$((PASSED + FAILED + WARNINGS))

echo "" | tee -a "$LOG_FILE"
log_info "Tests Passed:  $PASSED"
log_info "Tests Failed:  $FAILED"
log_info "Warnings:      $WARNINGS"
log_info "Total Checks:  $TOTAL"
echo "" | tee -a "$LOG_FILE"

# Calculate health score
if [ $TOTAL -gt 0 ]; then
    HEALTH_SCORE=$((PASSED * 100 / TOTAL))
else
    HEALTH_SCORE=0
fi

log_info "Health Score:  $HEALTH_SCORE%"
echo "" | tee -a "$LOG_FILE"

# Determine overall status
if [ $FAILED -eq 0 ]; then
    if [ $WARNINGS -eq 0 ]; then
        log "🎉 AUDIT RESULT: EXCELLENT - All systems operational!"
        AUDIT_STATUS="EXCELLENT"
    else
        log "✅ AUDIT RESULT: GOOD - Minor warnings detected"
        AUDIT_STATUS="GOOD"
    fi
else
    if [ $FAILED -lt 3 ]; then
        log "⚠️  AUDIT RESULT: FAIR - Some issues detected"
        AUDIT_STATUS="FAIR"
    else
        log "❌ AUDIT RESULT: POOR - Multiple failures detected"
        AUDIT_STATUS="POOR"
    fi
fi

echo "" | tee -a "$LOG_FILE"

# List critical issues
if [ ${#AUDIT_ISSUES[@]} -gt 0 ]; then
    log "🚨 CRITICAL ISSUES:"
    for issue in "${AUDIT_ISSUES[@]}"; do
        log "  → $issue"
    done
    echo "" | tee -a "$LOG_FILE"
fi

log "Log saved to: $LOG_FILE"
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Send Telegram notification if configured
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    # Prepare notification message
    if [ "$AUDIT_STATUS" = "EXCELLENT" ]; then
        emoji="🎉"
    elif [ "$AUDIT_STATUS" = "GOOD" ]; then
        emoji="✅"
    elif [ "$AUDIT_STATUS" = "FAIR" ]; then
        emoji="⚠️"
    else
        emoji="❌"
    fi
    
    message="$emoji *Ghost Daily Audit Report*
    
Status: *$AUDIT_STATUS* ($HEALTH_SCORE%)
Date: $(date +'%Y-%m-%d %H:%M:%S')

✅ Passed: $PASSED
❌ Failed: $FAILED
⚠️  Warnings: $WARNINGS

AI Memory: $(echo "$response" | jq -r '.components.ai_memory.records // "N/A"') decisions
Portfolio: $(echo "$response" | jq -r '.components.positions.wolf_qty // 0') shares"

    if [ ${#AUDIT_ISSUES[@]} -gt 0 ]; then
        message="$message

🚨 *Critical Issues:*"
        for issue in "${AUDIT_ISSUES[@]}"; do
            message="$message
→ $issue"
        done
    fi
    
    # Send via Telegram
    curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d "chat_id=$TELEGRAM_CHAT_ID" \
        -d "text=$message" \
        -d "parse_mode=Markdown" > /dev/null
    
    log "📱 Telegram notification sent"
fi

# Exit with appropriate code
if [ $FAILED -gt 0 ]; then
    exit 1
else
    exit 0
fi

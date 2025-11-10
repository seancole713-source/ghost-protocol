#!/bin/bash

# Ghost Trading System Monitor
# Checks health and restarts if needed

GHOST_URL="${GHOST_URL:-http://localhost:5000}"
LOG_FILE="/var/log/ghost-monitor.log"
MAX_FAILURES=3
FAILURE_COUNT=0

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_health() {
    if curl -f -s "$GHOST_URL/health" > /dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

restart_ghost() {
    log_message "🔄 Restarting Ghost..."
    
    if command -v docker-compose &> /dev/null; then
        cd /opt/GHOST
        docker-compose restart ghost
    elif command -v systemctl &> /dev/null; then
        systemctl restart ghost
    else
        pkill -f "uvicorn.*wolf_app"
        sleep 2
        cd /opt/GHOST
        nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 > /tmp/ghost.log 2>&1 &
    fi
    
    sleep 10
}

send_alert() {
    # Send alert via Telegram if configured
    if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
        MESSAGE="🚨 Ghost Health Alert: $1"
        curl -s -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
            -d "chat_id=$TELEGRAM_CHAT_ID" \
            -d "text=$MESSAGE" > /dev/null 2>&1
    fi
    
    log_message "$1"
}

# Main monitoring loop
while true; do
    if check_health; then
        if [ $FAILURE_COUNT -gt 0 ]; then
            log_message "✅ Ghost recovered after $FAILURE_COUNT failures"
            send_alert "Ghost is now healthy again"
        fi
        FAILURE_COUNT=0
    else
        FAILURE_COUNT=$((FAILURE_COUNT + 1))
        log_message "❌ Health check failed ($FAILURE_COUNT/$MAX_FAILURES)"
        
        if [ $FAILURE_COUNT -ge $MAX_FAILURES ]; then
            send_alert "Ghost is unhealthy. Attempting restart..."
            restart_ghost
            FAILURE_COUNT=0
            
            # Check if restart was successful
            sleep 15
            if check_health; then
                send_alert "Ghost successfully restarted"
            else
                send_alert "Ghost restart failed - manual intervention required"
            fi
        fi
    fi
    
    sleep 30
done

#!/bin/bash
# Test if Ghost is fully loaded with portfolio and watchlist

echo "🧪 Ghost System Test"
echo "===================="

# Test Health
echo "" 
echo "1. Health Check:"
health=$(curl -s http://localhost:5000/api/health | head -1)
if [ -n "$health" ]; then
    echo "   ✅ Server responding"
else
    echo "   ❌ Server not responding"
    exit 1
fi

# Test Portfolio
echo ""
echo "2. Portfolio:"
portfolio=$(curl -s http://localhost:5000/api/portfolio)
if echo "$portfolio" | grep -q "positions"; then
    pos_count=$(echo "$portfolio" | python3 -c "import sys, json; d=json.load(sys.stdin); print(len([p for p in d.get('positions', []) if p.get('qty', 0) > 0]))" 2>/dev/null || echo "0")
    echo "   Positions with qty > 0: $pos_count"
    if [ "$pos_count" -gt "0" ]; then
        echo "   ✅ Portfolio loaded"
    else
        echo "   ⚠️  Portfolio empty (checking STATE vs database)"
    fi
else
    echo "   ❌ Portfolio endpoint failed"
fi

# Test Watchlist
echo ""
echo "3. Watchlist:"
watchlist=$(curl -s http://localhost:5000/api/watchlist)
if echo "$watchlist" | grep -q "symbols"; then
    wl_count=$(echo "$watchlist" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('count', 0))" 2>/dev/null || echo "0")
    echo "   Symbol count: $wl_count"
    if [ "$wl_count" -gt "0" ]; then
        echo "   ✅ Watchlist loaded"
    else
        echo "   ❌ Watchlist empty"
    fi
else
    echo "   ❌ Watchlist endpoint failed"
fi

# Test Risk Status
echo ""
echo "4. Risk Status:"
risk=$(curl -s http://localhost:5000/api/risk/status)
if echo "$risk" | grep -q "can_trade"; then
    echo "   ✅ Risk endpoint working"
else
    echo "   ❌ Risk endpoint error (500)"
fi

# Check Telegram Config
echo ""
echo "5. Telegram:"
if [ -n "$TELEGRAM_BOT_TOKEN" ] && [ -n "$TELEGRAM_CHAT_ID" ]; then
    echo "   ✅ Telegram configured"
    if [ "$ALERT_SCHEDULE_OPEN_CLOSE" = "1" ]; then
        echo "   ✅ Market notifications enabled"
    else
        echo "   ⚠️  Market notifications disabled (set ALERT_SCHEDULE_OPEN_CLOSE=1)"
    fi
else
    echo "   ❌ Telegram not configured"
fi

echo ""
echo "===================="

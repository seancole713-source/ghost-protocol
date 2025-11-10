#!/bin/bash
# Test Ghost alerts and show detailed status

echo "🧪 Testing Ghost Alert System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if Telegram is configured
echo "1️⃣ Checking Telegram Configuration..."
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "   ❌ TELEGRAM_BOT_TOKEN not set"
    TG_CONFIGURED=false
else
    echo "   ✅ TELEGRAM_BOT_TOKEN is set"
    TG_CONFIGURED=true
fi

if [ -z "$TELEGRAM_CHAT_ID" ]; then
    echo "   ❌ TELEGRAM_CHAT_ID not set"
    TG_CONFIGURED=false
else
    echo "   ✅ TELEGRAM_CHAT_ID is set"
fi

echo ""

# Check selftest endpoint
echo "2️⃣ Testing /alerts/selftest..."
SELFTEST=$(curl -s http://localhost:5000/alerts/selftest)
SELFTEST_OK=$(echo "$SELFTEST" | jq -r '.ok // "false"')
echo "   Response: $SELFTEST"
if [ "$SELFTEST_OK" = "true" ]; then
    echo "   ✅ Alerts are configured"
else
    echo "   ❌ Alerts not configured"
fi

echo ""

# Send test alert
if [ "$TG_CONFIGURED" = true ]; then
    echo "3️⃣ Sending test alert..."
    RESPONSE=$(curl -s -X POST http://localhost:5000/alerts/test)
    echo "   Response: $RESPONSE"
    
    TEST_OK=$(echo "$RESPONSE" | jq -r '.ok // "false"')
    REASON=$(echo "$RESPONSE" | jq -r '.reason // "none"')
    
    if [ "$TEST_OK" = "true" ]; then
        echo "   ✅ Alert enqueued successfully"
        echo ""
        echo "   📱 CHECK YOUR TELEGRAM NOW!"
        echo "   You should receive a message with:"
        echo "   - Current WOLF price"
        echo "   - Your position"
        echo "   - Portfolio value"
        echo "   - Today's P&L"
        echo ""
        echo "   If you don't see it:"
        echo "   - Check that your bot token is correct"
        echo "   - Verify your chat ID is correct"
        echo "   - Check server logs: tail -100 ghost_server.log | grep telegram"
    else
        echo "   ❌ Alert failed"
        echo "   Reason: $REASON"
    fi
else
    echo "3️⃣ Skipping test send (Telegram not configured)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check recent Telegram activity in logs
echo "4️⃣ Checking recent Telegram activity in logs..."
echo ""
TELEGRAM_LOGS=$(tail -200 ghost_server.log | grep -i "telegram\|send_telegram" | tail -5)
if [ -z "$TELEGRAM_LOGS" ]; then
    echo "   ⚠️  No recent Telegram activity in logs"
    echo "   This could mean:"
    echo "   - Alert worker hasn't processed the queue yet"
    echo "   - Telegram sending failed silently"
    echo "   - Logs are missing send attempts"
else
    echo "   Recent activity:"
    echo "$TELEGRAM_LOGS" | sed 's/^/   /'
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check webhook info
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo "5️⃣ Telegram Webhook Info..."
    WEBHOOK_INFO=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo")
    WEBHOOK_URL=$(echo "$WEBHOOK_INFO" | jq -r '.result.url // "none"')
    PENDING=$(echo "$WEBHOOK_INFO" | jq -r '.result.pending_update_count // 0')
    
    echo "   Webhook URL: $WEBHOOK_URL"
    echo "   Pending updates: $PENDING"
    
    if [ "$WEBHOOK_URL" = "none" ] || [ "$WEBHOOK_URL" = "" ]; then
        echo "   ℹ️  No webhook set (using polling mode or no incoming messages)"
    elif [[ "$WEBHOOK_URL" == *"railway"* ]]; then
        echo "   ⚠️  Webhook points to Railway"
        echo "   Your Telegram messages go to Railway, not local server"
    elif [[ "$WEBHOOK_URL" == *"ngrok"* ]] || [[ "$WEBHOOK_URL" == *"localhost"* ]]; then
        echo "   ✅ Webhook points to local/tunnel"
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Summary:"
echo "--------"
if [ "$TG_CONFIGURED" = true ]; then
    echo "✅ Telegram is configured"
    echo "✅ Test alert was sent"
    echo "📱 Check your Telegram for the message!"
    echo ""
    echo "If you didn't receive it:"
    echo "1. Verify bot token: https://t.me/BotFather"
    echo "2. Check chat ID: curl \"https://api.telegram.org/bot\${TELEGRAM_BOT_TOKEN}/getUpdates\""
    echo "3. Try direct send: curl -X POST \"https://api.telegram.org/bot\${TELEGRAM_BOT_TOKEN}/sendMessage?chat_id=\${TELEGRAM_CHAT_ID}&text=Test\""
else
    echo "❌ Telegram is not configured"
    echo ""
    echo "To enable:"
    echo "1. Get bot token from @BotFather"
    echo "2. Set: export TELEGRAM_BOT_TOKEN='your-token'"
    echo "3. Set: export TELEGRAM_CHAT_ID='your-chat-id'"
    echo "4. Restart Ghost server"
fi

echo ""

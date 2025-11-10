#!/bin/bash
# Setup Telegram Webhook for Ghost on Railway

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📱 Telegram Webhook Setup for Ghost"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if token is provided
if [ -z "$1" ]; then
    echo "❌ Error: Please provide your Telegram bot token"
    echo ""
    echo "Usage:"
    echo "  ./setup_telegram_webhook.sh YOUR_BOT_TOKEN"
    echo ""
    echo "Example:"
    echo "  ./setup_telegram_webhook.sh 123456789:ABCdefGHIjklMNOpqrsTUVwxyz"
    echo ""
    echo "Get your token from:"
    echo "  Railway Dashboard → Variables → TELEGRAM_BOT_TOKEN"
    exit 1
fi

TOKEN="$1"
WEBHOOK_URL="https://web-production-8e9a0.up.railway.app/telegram/webhook"

echo "🔧 Setting webhook..."
echo "URL: $WEBHOOK_URL"
echo ""

# Set webhook
RESPONSE=$(curl -s "https://api.telegram.org/bot${TOKEN}/setWebhook?url=${WEBHOOK_URL}")

if echo "$RESPONSE" | grep -q '"ok":true'; then
    echo "✅ Webhook set successfully!"
    echo ""
    echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
else
    echo "❌ Failed to set webhook!"
    echo "$RESPONSE"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Verifying webhook info..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Get webhook info
INFO=$(curl -s "https://api.telegram.org/bot${TOKEN}/getWebhookInfo")
echo "$INFO" | python3 -m json.tool 2>/dev/null || echo "$INFO"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Setup Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Test your bot with these commands:"
echo "  /status  → Position & NAV"
echo "  /signal  → Trading signal"
echo "  /pnl     → Daily P&L (won/lost)"
echo "  /today   → Same as /pnl"
echo ""
echo "Ghost will now respond to your messages! 🚀"

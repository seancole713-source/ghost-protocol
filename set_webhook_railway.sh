#!/bin/bash
# Point Telegram webhook to Railway (where the code will be deployed)

set -e

echo "🔗 Configuring Telegram Webhook for Railway"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if TELEGRAM_BOT_TOKEN is set
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ ERROR: TELEGRAM_BOT_TOKEN not set"
    echo ""
    echo "Set your bot token:"
    echo "  export TELEGRAM_BOT_TOKEN='your-bot-token'"
    exit 1
fi

# Railway URL (update this if your Railway URL is different)
RAILWAY_URL="https://web-production-8e9a0.up.railway.app"

echo "🔧 Current webhook status:"
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo" | jq -r '.result | "URL: \(.url // "none")\nPending updates: \(.pending_update_count)"'
echo ""

echo "🗑️  Deleting old webhook..."
DELETE_RESULT=$(curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook")
DELETE_OK=$(echo "$DELETE_RESULT" | jq -r '.ok')
if [ "$DELETE_OK" = "true" ]; then
    echo "✅ Old webhook deleted"
else
    echo "⚠️  Delete webhook response: $DELETE_RESULT"
fi
echo ""

echo "🔗 Setting webhook to Railway: $RAILWAY_URL/telegram/webhook"
SET_RESULT=$(curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook?url=$RAILWAY_URL/telegram/webhook")
SET_OK=$(echo "$SET_RESULT" | jq -r '.ok')

if [ "$SET_OK" = "true" ]; then
    echo "✅ Webhook set successfully!"
else
    echo "❌ Failed to set webhook"
    echo "Response: $SET_RESULT"
    exit 1
fi
echo ""

echo "✅ New webhook status:"
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/getWebhookInfo" | jq '.result'
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Done!"
echo ""
echo "Your Telegram bot will now send messages to Railway."
echo ""
echo "⚠️  IMPORTANT: Make sure Railway has the new code deployed!"
echo ""
echo "To deploy to Railway:"
echo "  git add -A"
echo "  git commit -m 'feat: Add AI chat and fix test button'"
echo "  git push origin main"
echo ""
echo "Then set Railway environment variables:"
echo "  AGENTS_ENABLED=1"
echo "  AI_PROVIDER=openai"
echo "  AGENT_MODEL=gpt-4o-mini"
echo "  OPENAI_API_KEY=<your-key>"
echo ""

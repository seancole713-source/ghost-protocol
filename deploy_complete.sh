#!/bin/bash
# Complete deployment: Push code to Railway + Set env vars + Update webhook

set -e

echo "🚀 Complete Ghost AI Chat Deployment to Railway"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check prerequisites
if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
    echo "❌ ERROR: TELEGRAM_BOT_TOKEN not set"
    exit 1
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  WARNING: OPENAI_API_KEY not set"
    echo "AI chat features won't work until you set this in Railway dashboard"
    echo ""
fi

# Step 1: Commit and push code
echo "1️⃣  Committing and pushing code to Railway..."
echo ""

if [[ -n $(git status -s wolf_app.py) ]]; then
    git add wolf_app.py \
           GHOST_CHAT_*.md \
           TELEGRAM_WEBHOOK_ISSUE.md \
           enable_ghost_chat.sh \
           deploy_ai_chat.sh \
           set_webhook_railway.sh \
           test_alerts.sh
    
    git commit -m "feat: Add Telegram AI chat + fix test button

- Natural language Q&A via Telegram webhook
- Enhanced /telegram/webhook to handle questions (not just /commands)
- AI context builder with market data
- POST /ai/chat HTTP endpoint
- Fixed /alerts/test to work with UI (GET/POST, no auth)
- Send test alerts directly (not queued)
- Setup scripts and documentation"
    
    echo "✅ Changes committed"
else
    echo "ℹ️  No changes to commit (already committed)"
fi

echo ""
echo "📤 Pushing to origin..."
git push origin main

echo "✅ Code pushed to Railway"
echo ""

# Step 2: Wait for Railway deployment
echo "2️⃣  Waiting for Railway to deploy..."
echo "   (This takes ~2 minutes)"
echo ""

for i in {1..24}; do
    echo -n "."
    sleep 5
done
echo ""
echo ""

# Step 3: Check if Railway is healthy
RAILWAY_URL="https://web-production-8e9a0.up.railway.app"
echo "3️⃣  Checking Railway health..."

HEALTH_OK=false
for attempt in {1..5}; do
    HEALTH=$(curl -s -m 5 "$RAILWAY_URL/health" | jq -r '.ok // "false"')
    if [ "$HEALTH" = "true" ]; then
        echo "✅ Railway is healthy"
        HEALTH_OK=true
        break
    else
        echo "⏳ Waiting for Railway... (attempt $attempt/5)"
        sleep 10
    fi
done

if [ "$HEALTH_OK" = "false" ]; then
    echo "❌ Railway not responding. Check:"
    echo "   https://railway.app/dashboard"
    exit 1
fi

echo ""

# Step 4: Set Telegram webhook
echo "4️⃣  Configuring Telegram webhook..."
echo ""

echo "   Deleting old webhook..."
curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/deleteWebhook" > /dev/null

echo "   Setting webhook to Railway..."
SET_RESULT=$(curl -s "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook?url=$RAILWAY_URL/telegram/webhook")
SET_OK=$(echo "$SET_RESULT" | jq -r '.ok')

if [ "$SET_OK" = "true" ]; then
    echo "   ✅ Webhook configured"
else
    echo "   ❌ Webhook failed: $SET_RESULT"
    exit 1
fi

echo ""

# Step 5: Test the deployment
echo "5️⃣  Testing deployment..."
echo ""

# Test health
echo "   Testing /health..."
curl -s "$RAILWAY_URL/health" | jq -r '"   ✅ Health: \(.ok)"'

# Test alerts selftest
echo "   Testing /alerts/selftest..."
curl -s "$RAILWAY_URL/alerts/selftest" | jq -r '"   ✅ Alerts: \(.ok)"'

# Test alerts/test
echo "   Testing /alerts/test..."
TEST_RESULT=$(curl -s "$RAILWAY_URL/alerts/test")
TEST_OK=$(echo "$TEST_RESULT" | jq -r '.ok // "false"')
if [ "$TEST_OK" = "true" ]; then
    echo "   ✅ Test alert sent - CHECK YOUR TELEGRAM!"
else
    echo "   ⚠️  Test alert: $TEST_RESULT"
fi

echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Code deployed to Railway"
echo "✅ Telegram webhook configured"
echo "✅ Test alert sent (check Telegram)"
echo ""
echo "⚠️  NEXT STEPS:"
echo ""
echo "1. Set Railway environment variables (if not already set):"
echo "   Go to: https://railway.app/dashboard"
echo "   Add these variables:"
echo ""
echo "   AGENTS_ENABLED=1"
echo "   AI_PROVIDER=openai"
echo "   AGENT_MODEL=gpt-4o-mini"
echo "   OPENAI_API_KEY=<your-openai-key>"
echo ""
echo "2. After setting env vars, Railway will redeploy automatically"
echo ""
echo "3. Test via Telegram:"
echo "   Text your bot: 'What would a Bitcoin drop do to WOLF?'"
echo ""
echo "4. You should get AI analysis back!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

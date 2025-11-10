#!/bin/bash
# Deploy Ghost AI Chat to Railway
# This updates your Railway deployment with the new AI chat features

set -e

echo "🚀 Deploying Ghost AI Chat to Railway..."
echo ""

# Check if git has changes
if [[ -n $(git status -s) ]]; then
    echo "📝 Committing changes..."
    git add wolf_app.py GHOST_CHAT_*.md enable_ghost_chat.sh
    git commit -m "feat: Add Telegram AI chat capabilities

- Natural language Q&A via Telegram
- Enhanced webhook to handle questions (not just commands)
- AI context builder with market data
- /ai/chat HTTP endpoint
- Setup scripts and documentation"
    echo "✅ Changes committed"
else
    echo "✅ No changes to commit"
fi

echo ""
echo "📤 Pushing to Railway..."
git push origin main

echo ""
echo "⏳ Waiting for Railway deployment..."
echo "   Check status: https://railway.app/dashboard"
echo ""

# Wait a bit for deployment
sleep 10

echo "🔧 Setting Railway environment variables..."
echo ""
echo "Go to Railway dashboard and set:"
echo "  AGENTS_ENABLED=1"
echo "  AI_PROVIDER=openai"
echo "  AGENT_MODEL=gpt-4o-mini"
echo "  OPENAI_API_KEY=<your-key>"
echo ""
echo "Or use Railway CLI:"
echo "  railway variables set AGENTS_ENABLED=1"
echo "  railway variables set AI_PROVIDER=openai"
echo "  railway variables set AGENT_MODEL=gpt-4o-mini"
echo "  railway variables set OPENAI_API_KEY=<your-key>"
echo ""

read -p "Press Enter after setting Railway environment variables..."

echo ""
echo "🧪 Testing Railway deployment..."
RAILWAY_URL="https://web-production-8e9a0.up.railway.app"

# Test health
HEALTH=$(curl -s "${RAILWAY_URL}/health" | jq -r '.ok // "false"')
if [[ "$HEALTH" == "true" ]]; then
    echo "✅ Railway server is healthy"
else
    echo "❌ Railway server not responding"
    exit 1
fi

# Test AI chat (requires auth)
echo ""
echo "Testing AI chat endpoint..."
RESPONSE=$(curl -s -X POST "${RAILWAY_URL}/ai/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${GHOST_API_TOKEN}" \
  -d '{"question": "Hello Ghost"}' | jq -r '.answer // .detail // "Error"')

if [[ "$RESPONSE" == *"AI agent not enabled"* ]]; then
    echo "⚠️  AI not enabled yet on Railway. Set AGENTS_ENABLED=1"
elif [[ "$RESPONSE" == *"Error"* ]] || [[ "$RESPONSE" == *"detail"* ]]; then
    echo "❌ Error: $RESPONSE"
else
    echo "✅ AI is responding!"
    echo ""
    echo "Response: $RESPONSE"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Your Telegram bot will now respond to natural questions!"
echo "Test it: 'What would a Bitcoin drop do to WOLF?'"
echo ""

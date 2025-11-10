#!/bin/bash
# Quick fix: Set OpenAI API key in Railway

set -e

echo "🚀 GHOST OpenAI API Key Setup for Railway"
echo "=========================================="
echo ""

# Read the API key
OPENAI_KEY=$(grep "^OPENAI_AGENT_API_KEY=" /workspaces/GHOST/secrets.env | cut -d'=' -f2)

if [ -z "$OPENAI_KEY" ]; then
    echo "❌ ERROR: Could not find OPENAI_AGENT_API_KEY in secrets.env"
    exit 1
fi

echo "✅ Found API key: ${OPENAI_KEY:0:20}...${OPENAI_KEY: -4}"
echo ""
echo "📋 Copy the full API key below:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "$OPENAI_KEY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🔧 MANUAL SETUP STEPS:"
echo ""
echo "1. Go to Railway Dashboard:"
echo "   https://railway.app/dashboard"
echo ""
echo "2. Find your GHOST project (web-production-8e9a0)"
echo ""
echo "3. Click on the Variables tab"
echo ""
echo "4. Add or update this variable:"
echo "   Variable name: OPENAI_AGENT_API_KEY"
echo "   Variable value: [paste the key from above]"
echo ""
echo "5. Railway will automatically redeploy (takes ~2 minutes)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🧪 VERIFICATION:"
echo "After Railway redeploys, test with:"
echo "  curl -X POST https://web-production-8e9a0.up.railway.app/telegram/webhook \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"message\":{\"chat\":{\"id\":123},\"text\":\"test\"}}'"
echo ""
echo "Or send a Telegram message: \"What's today prediction\""
echo ""

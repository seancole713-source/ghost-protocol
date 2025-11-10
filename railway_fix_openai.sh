#!/bin/bash
# Direct Railway OpenAI API key setup

set -e

OPENAI_KEY=$(grep "^OPENAI_AGENT_API_KEY=" /workspaces/GHOST/secrets.env | cut -d'=' -f2)

if [ -z "$OPENAI_KEY" ]; then
    echo "❌ ERROR: Could not find OPENAI_AGENT_API_KEY"
    exit 1
fi

echo "🚀 Setting OpenAI API key in Railway project: tender-benevolence"
echo ""

cd /workspaces/GHOST

# Link to project
echo "📡 Linking to Railway project..."
railway link --project f910dbba-dc10-4a8b-b654-28001e64f4ec --environment production --service web

echo ""
echo "🔧 Setting OPENAI_AGENT_API_KEY variable..."
railway variables --set "OPENAI_AGENT_API_KEY=$OPENAI_KEY"

echo ""
echo "✅ Done! Railway will redeploy automatically."
echo ""
echo "⏳ Wait 1-2 minutes for deployment, then test with:"
echo "   Send Telegram message: \"What's today prediction\""
echo ""

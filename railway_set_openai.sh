#!/bin/bash
# Automated Railway OpenAI API key setup

set -e

echo "🚀 Setting OpenAI API key in Railway..."

# Read the API key
OPENAI_KEY=$(grep "^OPENAI_AGENT_API_KEY=" /workspaces/GHOST/secrets.env | cut -d'=' -f2)

if [ -z "$OPENAI_KEY" ]; then
    echo "❌ ERROR: Could not find OPENAI_AGENT_API_KEY in secrets.env"
    exit 1
fi

echo "✅ Found API key: ${OPENAI_KEY:0:20}...${OPENAI_KEY: -4}"
echo ""

# Create a railway project config
cd /workspaces/GHOST

echo "📡 Linking to Railway project..."
# Use the project ID from the URL: web-production-8e9a0.up.railway.app
# The project name is tender-benevolence

# Try to link using project name (interactive, may need manual intervention)
# railway link tender-benevolence 2>&1 || echo "Project already linked or manual link needed"

echo ""
echo "🔧 Setting OPENAI_AGENT_API_KEY variable..."
railway variables --set "OPENAI_AGENT_API_KEY=$OPENAI_KEY"

echo ""
echo "✅ Variable set! Railway will now redeploy automatically."
echo ""
echo "⏳ Waiting for deployment (this may take 1-2 minutes)..."
echo ""
echo "You can check deployment status at:"
echo "  https://railway.app/project"
echo ""
echo "🧪 After deployment completes, test with Telegram:"
echo "  Send message: \"What's today prediction\""
echo ""

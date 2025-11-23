#!/bin/bash
# Fix OpenAI API key in Railway deployment

echo "🔧 Setting OpenAI API key in Railway..."

# Read the API key from secrets.env
OPENAI_KEY=$(grep "^OPENAI_AGENT_API_KEY=" /workspaces/GHOST/secrets.env | cut -d'=' -f2)

if [ -z "$OPENAI_KEY" ]; then
    echo "❌ ERROR: Could not read OPENAI_AGENT_API_KEY from secrets.env"
    exit 1
fi

echo "✅ Found API key: ${OPENAI_KEY:0:20}..."

# Note: Railway CLI needs to be linked to the project first
# If you get "No linked project found", run: railway link

# Try to set the variable directly via Railway API or CLI
echo ""
echo "📝 To set this in Railway, run ONE of these commands:"
echo ""
echo "Option 1 - Using Railway CLI (if project is linked):"
echo "  railway variables set OPENAI_AGENT_API_KEY=\"$OPENAI_KEY\""
echo ""
echo "Option 2 - Using Railway Dashboard:"
echo "  1. Go to https://railway.app/project"
echo "  2. Select your GHOST project"
echo "  3. Go to Variables tab"
echo "  4. Add/Update: OPENAI_AGENT_API_KEY = $OPENAI_KEY"
echo ""
echo "After setting the variable, Railway will automatically redeploy."
echo ""
echo "🧪 Testing current key validity..."
TEST_RESPONSE=$(curl -s -w "%{http_code}" -o /tmp/openai_validate.json https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_KEY" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"test"}],"max_tokens":5}')

if [ "$TEST_RESPONSE" = "200" ]; then
    echo "✅ OpenAI API key is valid!"
    cat /tmp/openai_validate.json | jq -r '.choices[0].message.content'
else
    echo "❌ OpenAI API key test failed with status: $TEST_RESPONSE"
    cat /tmp/openai_validate.json
fi

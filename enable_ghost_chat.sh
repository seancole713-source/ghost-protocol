#!/bin/bash
# Enable Ghost AI Chat capabilities
# Usage: ./enable_ghost_chat.sh [openai|ollama]

set -e

PROVIDER=${1:-openai}

echo "🤖 Enabling Ghost AI Chat..."
echo ""

# Check if OpenAI key is set
if [ "$PROVIDER" = "openai" ]; then
    if [ -z "$OPENAI_API_KEY" ]; then
        echo "❌ ERROR: OPENAI_API_KEY not set"
        echo ""
        echo "Set your OpenAI API key:"
        if command -v railway >/dev/null 2>&1; then
            echo "  export OPENAI_API_KEY=\"$(railway variables get OPENAI_API_KEY)\""
        else
            echo "  export OPENAI_API_KEY='<paste the live key from the OpenAI dashboard>'"
        fi
        echo ""
        exit 1
    fi
    echo "✅ OpenAI API key found"
    export AI_PROVIDER=openai
    export AGENT_MODEL=${AGENT_MODEL:-gpt-4o-mini}
    echo "✅ Using model: $AGENT_MODEL"
elif [ "$PROVIDER" = "ollama" ]; then
    # Check if Ollama is running
    if ! curl -s http://localhost:11434/api/tags >/dev/null 2>&1; then
        echo "❌ ERROR: Ollama not running"
        echo ""
        echo "Start Ollama first:"
        echo "  ollama serve"
        echo ""
        exit 1
    fi
    echo "✅ Ollama found"
    export AI_PROVIDER=ollama
    export AGENT_MODEL=${AGENT_MODEL:-llama3.2}
    export OLLAMA_BASE_URL=http://localhost:11434/v1
    echo "✅ Using model: $AGENT_MODEL"
else
    echo "❌ ERROR: Unknown provider '$PROVIDER'"
    echo "Usage: $0 [openai|ollama]"
    exit 1
fi

# Enable agents
export AGENTS_ENABLED=1
echo "✅ AI agents enabled"
echo ""

# Restart Ghost server
echo "🔄 Restarting Ghost server..."
pkill -f "uvicorn.*wolf_app" 2>/dev/null || true
sleep 2

cd /workspaces/GHOST
source .venv/bin/activate

# Start with environment variables
AGENTS_ENABLED=1 \
AI_PROVIDER="$AI_PROVIDER" \
AGENT_MODEL="$AGENT_MODEL" \
OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-}" \
nohup python -m uvicorn wolf_app:APP --host 0.0.0.0 --port 5000 --reload > ghost_server.log 2>&1 &

PID=$!
echo "✅ Ghost server started (PID: $PID)"
echo ""

# Wait for server to be ready
echo "⏳ Waiting for server to start..."
for i in {1..20}; do
    if curl -s -m 1 http://localhost:5000/health >/dev/null 2>&1; then
        echo "✅ Server is ready!"
        echo ""
        break
    fi
    sleep 1
done

# Test the chat endpoint
echo "🧪 Testing AI chat..."
echo ""
RESPONSE=$(curl -s -X POST http://localhost:5000/ai/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${GHOST_API_TOKEN}" \
  -d '{"question": "Hello Ghost, are you working?"}' | jq -r '.answer // .detail // "Error"')

if [[ "$RESPONSE" == *"AI agent not enabled"* ]]; then
    echo "❌ AI still not enabled. Check logs:"
    echo "   tail -50 ghost_server.log"
    exit 1
elif [[ "$RESPONSE" == *"missing bearer token"* ]]; then
    echo "⚠️  Auth required. Set GHOST_API_TOKEN:"
    echo "   export GHOST_API_TOKEN='your-token'"
    exit 1
else
    echo "🎉 SUCCESS! Ghost AI is responding:"
    echo ""
    echo "$RESPONSE"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🤖 Ghost AI Chat is READY!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📱 Via Telegram:"
echo "   Just text your bot any question!"
echo "   Example: 'What would a Bitcoin drop do to WOLF?'"
echo ""
echo "🌐 Via HTTP:"
echo "   curl -X POST http://localhost:5000/ai/chat \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -H 'Authorization: Bearer ${GHOST_API_TOKEN}' \\"
echo "     -d '{\"question\": \"Your question here\"}'"
echo ""
echo "🎯 Commands:"
echo "   /status - Portfolio status"
echo "   /signal - Trading signal"
echo "   /pnl    - Daily P&L"
echo "   /help   - Show help"
echo ""
echo "💰 Cost: ~$0.0001 per question (with gpt-4o-mini)"
echo ""

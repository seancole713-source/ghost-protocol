#!/bin/bash
# Quick deployment check script

echo "🔍 Checking Ghost Protocol Deployment..."
echo ""

# Health check
echo "1. Health Endpoint:"
curl -s https://ghost-sniper-bot-seancole713-production.up.railway.app/ui/health | python3 -m json.tool 2>/dev/null || echo "❌ Not responding (might still be building)"
echo ""

# World context
echo "2. World Context Endpoint:"
curl -s https://ghost-sniper-bot-seancole713-production.up.railway.app/api/world/context | python3 -m json.tool 2>/dev/null | head -20 || echo "❌ Not responding"
echo ""

echo "✅ If you see JSON above, deployment is working!"
echo "❌ If you see 502 errors, wait 1-2 more minutes for build to complete"

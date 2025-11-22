#!/bin/bash
# Railway Environment Variable Cleanup
# These variables are safe to delete (unused/duplicates)

echo "🧹 Cleaning up unused Railway environment variables..."
echo ""

# Delete unused placeholders
railway variables delete AGENT_ROLE
railway variables delete AGENT_POLICY
railway variables delete AGENT_RUN_INTERVAL_SEC
railway variables delete MEMORY_TTL_DAYS
railway variables delete VECTOR_SOURCE
railway variables delete VECTOR_STORE_ID
railway variables delete OPENAI_ORG_ID
railway variables delete CACHE_MODE
railway variables delete CACHE_TTL
railway variables delete AUTO_FIXER_ENABLED
railway variables delete AUTO_FIX_INTERVAL_SEC
railway variables delete AUTO_RESTART_COOLDOWN_SEC
railway variables delete DATA_FRESHNESS_SEC
railway variables delete ALERT_CHANNEL
railway variables delete AGENT_ENDPOINT_URL
railway variables delete ANTHROPIC_API_KEY

echo ""
echo "✅ Cleanup complete!"
echo ""
echo "📝 Optional: Add these for Ghost V3 features:"
echo "railway variables set RESEARCH_LLM_ON=1"
echo "railway variables set RESEARCH_LLM_MODEL=gpt-4o-mini"
echo "railway variables set WOLF_QTY=8.41959051"
echo "railway variables set WOLF_AVG_COST=359.28"

#!/bin/bash
# Ghost Cockpit Live Restore - Railway Environment Configuration
# Mission: Enforce production environment for 100% live operation
# Date: 2025-11-10

echo "=========================================="
echo "PHASE 1: Railway Environment Configuration"
echo "=========================================="

# Critical Price Provider Settings
railway variables set STOCK_PRICE_SOURCE=polygon
railway variables set PRICE_YAHOO_FIRST=0
railway variables set PRICE_PROVIDER_TIMEOUT_S=1.5
railway variables set PRICE_PROVIDER_TIMEOUT=1.5
railway variables set DATA_FRESHNESS_SEC=60
railway variables set PRICE_MIN_PROVIDERS=1
railway variables set PRICE_REQUIRE_QUORUM=0

# Stock Trading Configuration
railway variables set FOCUS_WOLF_ONLY=0
railway variables set STOCKS_ENABLED=1
railway variables set PREDICT_STOCKS_ENABLED=1
railway variables set SIM_MODE=0

# Crypto Configuration
railway variables set CRYPTO_ENABLED=1

# Price Seeding Configuration
railway variables set ALLOW_SAFE_PRICE=0
railway variables set ALLOW_SEEDED_PRICE=1
railway variables set SEEDED_PRICE_MAX_AGE_S=900

# Timezone
railway variables set GHOST_TZ=America/Chicago

# Server Configuration
railway variables set UVICORN_TIMEOUT_KEEP_ALIVE=75
railway variables set UVICORN_LIMIT_MAX_REQUESTS=10000

echo ""
echo "✅ Environment variables configured"
echo ""
echo "⚠️  VERIFY THESE ARE SET:"
echo "   - POLYGON_API_KEY"
echo "   - ALPHAVANTAGE_API_KEY"
echo "   - GHOST_API_TOKEN"
echo ""
echo "Run: railway variables list | grep -E 'POLYGON|ALPHAVANTAGE|GHOST_API_TOKEN'"
echo ""

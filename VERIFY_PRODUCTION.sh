#!/bin/bash
# PRODUCTION VERIFICATION SCRIPT
# Run this in Railway production to verify news brain integration

echo "=========================================="
echo "GHOST NEWS BRAIN - PRODUCTION VERIFICATION"
echo "=========================================="
echo ""
echo "Deployment time: $(date)"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Test 1: Check Ghost News Brain is running
echo "TEST 1: Ghost News Brain Status"
echo "------------------------------------------"
python3 <<'EOF'
import sys
sys.path.insert(0, '/app')

from core.intelligence.ghost_news_brain import get_news_brain

brain = get_news_brain()
status = brain.get_status()

print(f"✅ Ghost News Brain initialized")
print(f"   Anthropic available: {status['anthropic_available']}")
print(f"   API key present: {status['api_key_present']}")
print(f"   Enabled: {status['enabled']}")
print(f"   RSS feeds: {status['rss_feeds_count']}")
print(f"   DB configured: {status['db_configured']}")
print()

if not status['enabled']:
    print("⚠️  WARNING: Ghost News Brain is NOT enabled!")
    print("   Anthropic client not available")
    print()
EOF

# Test 2: Check news_analysis table
echo "TEST 2: News Analysis Table (Recent Records)"
echo "------------------------------------------"
python3 <<'EOF'
import sys
sys.path.insert(0, '/app')
import os
from datetime import datetime, timedelta

db_url = os.getenv("DATABASE_URL")
if not db_url:
    print("❌ DATABASE_URL not set")
    sys.exit(1)

if db_url.startswith("postgresql"):
    import psycopg2
    import psycopg2.extras
    
    try:
        conn = psycopg2.connect(db_url)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        
        # Check if table exists
        cur.execute("SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'news_analysis')")
        exists = cur.fetchone()['exists']
        
        if not exists:
            print("❌ news_analysis table does NOT exist")
            print("   Ghost News Brain may not have run yet")
            conn.close()
            sys.exit(1)
        
        # Get recent records
        cur.execute("""
            SELECT 
                analysis_id,
                analysis_time,
                headlines_fetched,
                events_found,
                predictions_affected,
                EXTRACT(EPOCH FROM (NOW() - analysis_time))/60 as age_minutes
            FROM news_analysis 
            ORDER BY analysis_time DESC 
            LIMIT 5
        """)
        
        records = cur.fetchall()
        conn.close()
        
        if not records:
            print("⚠️  news_analysis table is EMPTY")
            print("   Ghost News Brain loop may not be running")
            print("   Check: NEWS_ANALYSIS_ENABLED env var")
            sys.exit(1)
        
        print(f"✅ Found {len(records)} recent analysis records:")
        for r in records:
            age = r['age_minutes']
            status = "🟢" if age < 30 else "🟡" if age < 60 else "🔴"
            print(f"{status} {r['analysis_time']}: {r['headlines_fetched']} headlines, {r['events_found']} events ({age:.0f}m ago)")
        
        # Check if data is fresh
        latest = records[0]
        if latest['age_minutes'] > 60:
            print(f"\n⚠️  WARNING: Latest analysis is {latest['age_minutes']:.0f} minutes old")
            print("   Ghost News Brain may not be running")
        else:
            print(f"\n✅ Data is fresh (last run: {latest['age_minutes']:.0f}m ago)")
        
        print()
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        sys.exit(1)
else:
    print("⚠️  Non-PostgreSQL database, skipping table check")
    print()
EOF

# Test 3: Test sentiment engine with real symbol
echo "TEST 3: Sentiment Engine (Real Symbol)"
echo "------------------------------------------"
python3 <<'EOF'
import sys
sys.path.insert(0, '/app')

from core.data_pillars.sentiment_engine import SentimentEngine

engine = SentimentEngine()
response = engine.get_signals("BTC")

print(f"Symbol: BTC")
print(f"Signals: {response.signal_count()}")
print(f"Available: {response.available_signal_count()}")
print(f"Execution time: {response.execution_time_ms}ms")
print()

for signal in response.signals:
    available = "✅" if signal.data_available else "❌"
    print(f"{available} {signal.name}: {signal.value} (source: {signal.source})")

print()

# Check source
sources = [s.source for s in response.signals if s.data_available]
if 'ghost_news_brain' in sources:
    print("✅ Using Ghost News Brain cached analysis")
elif 'rss_scan' in sources:
    print("✅ Using RSS feed scan")
elif 'no_news_neutral' in sources:
    print("ℹ️  No recent news for BTC (neutral)")
else:
    print("⚠️  Unknown source")

print()
EOF

# Test 4: Test world context (SPY/VIX)
echo "TEST 4: World Context (SPY/VIX)"
echo "------------------------------------------"
python3 <<'EOF'
import sys
sys.path.insert(0, '/app')

from core.world_context import get_world_context

context = get_world_context()

spy_price = context["spy"]["price"]
spy_provider = context["spy"]["provider"]
vix_level = context["vix"]["level"]
vix_status = context["vix"]["status"]
market_mood = context["market_mood"]["sentiment"]

if spy_price:
    print(f"✅ SPY: ${spy_price} (provider: {spy_provider})")
else:
    print(f"❌ SPY: NULL - Both price_quorum and yfinance failed")

if vix_level:
    print(f"✅ VIX: {vix_level} (status: {vix_status})")
else:
    print(f"❌ VIX: NULL - Both price_quorum and yfinance failed")

print(f"   Market Mood: {market_mood} (score: {context['market_mood']['score']})")
print()

if not spy_price or not vix_level:
    print("⚠️  World context failing - check price providers")
    print()
EOF

# Test 5: Feature orchestrator full test
echo "TEST 5: Feature Orchestrator (Full Integration)"
echo "------------------------------------------"
python3 <<'EOF'
import sys
sys.path.insert(0, '/app')

from core.data_pillars.feature_orchestrator import get_feature_orchestrator

orchestrator = get_feature_orchestrator()
features = orchestrator.get_all_features("RNDR")

print(f"Symbol: RNDR")
print(f"Total features: {features['feature_count']}")
print(f"Available: {features['available_count']}")
print(f"Execution time: {features['execution_time_ms']}ms")
print()

print("Pillar Status:")
for pillar, stat in features['feature_availability'].items():
    if "DISABLED" in stat or "0/0" in stat:
        status = "❌"
    elif "/" in stat:
        parts = stat.split("/")
        if parts[0] == parts[1]:
            status = "✅"
        else:
            status = "🟡"
    else:
        status = "❓"
    print(f"  {status} {pillar}: {stat}")

print()

# Check key features
key_features = [
    "NEWS_SENTIMENT_SCORE",
    "NEWS_COUNT_24H", 
    "SPY_PRICE",
    "VIX_LEVEL",
    "MARKET_REGIME"
]

print("Key Features:")
for key in key_features:
    value = features['features'].get(key)
    if value is not None:
        print(f"  ✅ {key}: {value}")
    else:
        print(f"  ❌ {key}: None")

print()
EOF

# Summary
echo "=========================================="
echo "SUMMARY"
echo "=========================================="
echo ""
echo "✅ PASS = Feature working with real data"
echo "⚠️  WARN = Feature exists but may have issues"  
echo "❌ FAIL = Feature not working"
echo ""
echo "Next steps if any failures:"
echo "1. Check Railway logs: railway logs --tail 200"
echo "2. Check environment variables: railway variables"
echo "3. Verify Ghost News Brain loop is running"
echo "4. Check database connections"
echo ""

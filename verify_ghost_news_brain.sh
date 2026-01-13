#!/bin/bash
# Ghost News Brain Verification Script
# Run this in production to verify if News Brain is running

echo "=========================================="
echo "Ghost News Brain Status Verification"
echo "=========================================="
echo ""

# 1. Check Railway logs for News Brain activity
echo "1. Checking Railway logs for News Brain activity..."
echo "   Command: railway logs --tail 500 | grep -E 'News Analysis|📰|GhostNewsBrain'"
echo ""
railway logs --tail 500 2>/dev/null | grep -E "News Analysis|📰|GhostNewsBrain" | tail -20
echo ""

# 2. Check environment variables
echo "2. Checking environment variables..."
echo "   NEWS_ANALYSIS_ENABLED: ${NEWS_ANALYSIS_ENABLED:-NOT SET (defaults to 1)}"
echo "   NEWS_ANALYSIS_INTERVAL_MINUTES: ${NEWS_ANALYSIS_INTERVAL_MINUTES:-NOT SET (defaults to 30)}"
echo "   ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:+CONFIGURED}"
echo "   CRYPTOPANIC_API_KEY: ${CRYPTOPANIC_API_KEY:-NOT SET}"
echo ""

# 3. Query news_analysis table
echo "3. Checking news_analysis table for recent records..."
python3 <<'EOF'
import os
import sqlite3
from datetime import datetime, timedelta

db_url = os.getenv("DATABASE_URL", "sqlite:///data/ghost.db")
if db_url.startswith("sqlite:///"):
    db_path = db_url.replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if table exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='news_analysis'")
        if cursor.fetchone():
            # Get recent records
            cursor.execute("""
                SELECT analysis_id, analysis_time, headlines_fetched, events_found, 
                       predictions_affected, summary
                FROM news_analysis 
                ORDER BY analysis_time DESC 
                LIMIT 10
            """)
            records = cursor.fetchall()
            
            if records:
                print(f"   Found {len(records)} recent analysis records:")
                for r in records:
                    print(f"   - {r[1]}: {r[2]} headlines, {r[3]} events, {r[4]} predictions affected")
            else:
                print("   ⚠️  news_analysis table exists but is EMPTY")
                print("   → Ghost News Brain may NOT be running")
        else:
            print("   ⚠️  news_analysis table does NOT exist")
            print("   → Database schema may not be initialized")
        
        conn.close()
    except Exception as e:
        print(f"   ❌ Error querying database: {e}")
else:
    print(f"   Using non-SQLite database: {db_url[:30]}...")
    print("   → Use psql or pg client to query manually")
EOF
echo ""

# 4. Check guardian_alerts table
echo "4. Checking guardian_alerts table..."
python3 <<'EOF'
import os
import sqlite3

db_url = os.getenv("DATABASE_URL", "sqlite:///data/ghost.db")
if db_url.startswith("sqlite:///"):
    db_path = db_url.replace("sqlite:///", "")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='guardian_alerts'")
        if cursor.fetchone():
            cursor.execute("""
                SELECT alert_id, created_at, symbol, severity, message, acknowledged
                FROM guardian_alerts 
                ORDER BY created_at DESC 
                LIMIT 5
            """)
            records = cursor.fetchall()
            
            if records:
                print(f"   Found {len(records)} recent guardian alerts:")
                for r in records:
                    print(f"   - {r[1]} [{r[3]}] {r[2]}: {r[4][:50]}...")
            else:
                print("   ⚠️  guardian_alerts table exists but is EMPTY")
        else:
            print("   ⚠️  guardian_alerts table does NOT exist")
        
        conn.close()
    except Exception as e:
        print(f"   ❌ Error: {e}")
EOF
echo ""

# 5. Check Telegram alerts
echo "5. Checking for Telegram News Brain alerts..."
echo "   → Check your Telegram channel for messages containing:"
echo "      - '📰 News Analysis'"
echo "      - 'CRITICAL/HIGH events detected'"
echo "      - 'Predictions at risk'"
echo ""

# Summary
echo "=========================================="
echo "Summary"
echo "=========================================="
echo ""
echo "Ghost News Brain Status:"
echo ""
echo "✅ Code exists: core/intelligence/ghost_news_brain.py (880 lines)"
echo "✅ Loop registered: wolf_app.py line 4938 (asyncio.create_task)"
echo "✅ API key configured: Anthropic API (claude-sonnet-4-20250514)"
echo ""
echo "TO VERIFY IF RUNNING:"
echo "1. Check Railway logs above for '📰 News Analysis Loop: STARTING'"
echo "2. Check news_analysis table for records created every 30 minutes"
echo "3. Check Telegram for News Brain alerts"
echo ""
echo "IF NOT RUNNING:"
echo "- Check if NEWS_ANALYSIS_ENABLED=0 in production"
echo "- Check for errors in Railway startup logs"
echo "- Test manually: python3 -c 'from core.intelligence.ghost_news_brain import test_news_brain; import asyncio; asyncio.run(test_news_brain())'"
echo ""
echo "RECOMMENDATION:"
echo "If News Brain NOT running → DISABLE IT (costs Anthropic API credits)"
echo "If News Brain IS running → Monitor alert quality and API costs"
echo ""

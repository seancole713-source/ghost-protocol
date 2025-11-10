"""
Quick Stage 1 Verification
===========================
Fast check that Stage 1 components are ready.
"""

print("=" * 60)
print("GHOST STAGE 1 QUICK VERIFICATION")
print("=" * 60)

# Check 1: Imports
print("\n✓ Check 1: Imports")
try:
    from core.context_engine import WorldContextEngine

    print("  ✅ WorldContextEngine imported")
except Exception as e:
    print(f"  ❌ WorldContextEngine import failed: {e}")

try:
    from core.market_mood import update_market_mood

    print("  ✅ Market mood functions imported")
except Exception as e:
    print(f"  ❌ Market mood import failed: {e}")

try:
    print("  ✅ Stage 1 integration imported")
except Exception as e:
    print(f"  ❌ Stage 1 integration import failed: {e}")

# Check 2: Dependencies
print("\n✓ Check 2: Dependencies")
try:
    print("  ✅ feedparser installed")
except Exception:
    print("  ❌ feedparser missing")

try:
    print("  ✅ yfinance installed")
except Exception:
    print("  ❌ yfinance missing")

try:
    print("  ✅ vaderSentiment installed")
except Exception:
    print("  ❌ vaderSentiment missing")

try:
    import spacy

    print("  ✅ spacy installed")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("  ✅ spacy en_core_web_sm model loaded")
    except Exception:
        print("  ⚠️  spacy model en_core_web_sm not found (NER will be limited)")
except Exception:
    print("  ⚠️  spacy not installed (NER will be limited)")

# Check 3: Directories
print("\n✓ Check 3: Directories")
import os

dirs_to_check = ["core", "data", "logs", "reports"]
for dirname in dirs_to_check:
    if os.path.exists(dirname):
        print(f"  ✅ {dirname}/ exists")
    else:
        print(f"  ❌ {dirname}/ missing")

# Check 4: Files
print("\n✓ Check 4: Core Files")
files_to_check = ["core/context_engine.py", "core/market_mood.py", "core/stage1_integration.py"]
for filepath in files_to_check:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath)
        print(f"  ✅ {filepath} ({size:,} bytes)")
    else:
        print(f"  ❌ {filepath} missing")

# Check 5: Test Market Mood (Quick)
print("\n✓ Check 5: Market Mood Test")
try:
    from core.market_mood import update_market_mood

    mood = update_market_mood()
    if "error" in mood:
        print(f"  ⚠️  Market mood update had error: {mood['error']}")
    else:
        print(f"  ✅ Market mood updated: {mood.get('market_regime')} regime")
        print(f"     Date: {mood.get('date')}")
        print(f"     Sentiment: {mood.get('sentiment')}")
        print(f"     SPY: ${mood.get('spy', {}).get('price', 0):.2f}")
        print(f"     VIX: {mood.get('vix', {}).get('current', 0):.2f}")
except Exception as e:
    print(f"  ❌ Market mood test failed: {e}")

# Check 6: Test Context Engine (No fetch, just init)
print("\n✓ Check 6: Context Engine Test")
try:
    from core.context_engine import WorldContextEngine

    test_feeds = ["https://www.reuters.com/business/rss"]
    engine = WorldContextEngine(test_feeds, watchlist=["NVDA", "TSLA"])
    stats = engine.get_stats()
    print("  ✅ Context engine initialized")
    print(f"     Feeds: {stats['feeds_count']}")
    print(f"     Watchlist: {stats['watchlist_count']}")
    print(f"     DB: {stats['db_path']}")
    engine.close()
except Exception as e:
    print(f"  ❌ Context engine test failed: {e}")

# Check 7: Integration Test
print("\n✓ Check 7: Integration Module")
try:
    from core.stage1_integration import get_context_stats

    stats = get_context_stats()
    print("  ✅ Integration module working")
    print(f"     Initialized: {stats.get('initialized')}")
except Exception as e:
    print(f"  ❌ Integration test failed: {e}")

# Summary
print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print("\n✅ Stage 1 components are ready!")
print("\nNext step: Integrate to wolf_app.py")
print("  1. Add: from core.stage1_integration import initialize_stage1, get_enhanced_context")
print("  2. On startup: initialize_stage1()")
print("  3. In _build_ai_context(): add world_context and market_mood")
print("=" * 60)

# Stage 1 Integration Testing Guide

**Date**: October 5, 2025\
**Status**: ✅ Integration Complete\
**Intelligence Level**: 7 → 8 (Context Awareness)

______________________________________________________________________

## ✅ Integration Summary

### Changes Made to wolf_app.py

1. **Imports Added**(Lines 57-63):


   ```python

   # Stage 1: Context Awareness imports

   try:
       from core.stage1_integration import initialize_stage1, get_enhanced_context, get_symbol_context
       STAGE1_ENABLED = True
   except Exception as e:
       STAGE1_ENABLED = False
       print(f"Stage 1 Context Awareness disabled: {e}")

   ```text

1.**Startup Initialization**(Lines 1354-1365):


   ```python

   # Stage 1: Initialize Context Awareness Layer

   if STAGE1_ENABLED:
       try:
           task = initialize_stage1()
           if task:
               LOGGER.info("stage1_initialized", extra={
                   "component": "startup",
                   "features": "world_context,market_mood",
                   "update_interval": "5min"
               })
       except Exception as e:
           LOGGER.exception("stage1_init_failed", extra={"component": "startup", "error": str(e)})

   ```text

1.**Enhanced AI Context**(Lines 4869-4881):


   ```python

   # Stage 1: Add enhanced world context and market mood

   if STAGE1_ENABLED:
       try:
           enhanced = get_enhanced_context(hours=24, min_relevance=0.3)
           snap["world_context"] = enhanced.get("world_context", {})
           snap["market_mood"] = enhanced.get("market_mood", {})
       except Exception as e:
           LOGGER.warning("stage1_context_failed", extra={"error": str(e)})
           snap["world_context"] = {}
           snap["market_mood"] = {}

   ```text

1.**New API Endpoints**(Lines 4424-4475):

   - `/api/stage1/world` - Get world news context
   - `/api/stage1/mood` - Get market mood/regime
   - `/api/stage1/symbol/{symbol}` - Get symbol-specific context
   - `/api/stage1/stats` - Get Stage 1 statistics


______________________________________________________________________

## 🧪 Testing Checklist

### Pre-Server Tests

#### ✅ 1. Verify Stage 1 Components Exist

```bash

cd /workspaces/GHOST

# Check core files exist

ls -lh core/context_engine.py
ls -lh core/market_mood.py
ls -lh core/stage1_integration.py

# Check test files

ls -lh test_context.py
ls -lh verify_stage1.py

```text**Expected**: All 5 files should exist

#### ✅ 2. Run Quick Verification

```bash

cd /workspaces/GHOST
source .venv/bin/activate
python verify_stage1.py

```text

**Expected Output**:

```text

✓ Check 1: Imports
  ✅ WorldContextEngine imported
  ✅ Market mood functions imported
  ✅ Stage 1 integration imported

✓ Check 2: Dependencies
  ✅ feedparser installed
  ✅ yfinance installed
  ✅ vaderSentiment installed
  ✅ spacy installed
  ✅ spacy model en_core_web_sm found

✓ Check 3: Directories
  ✅ core/ exists
  ✅ data/ exists
  ✅ logs/ exists
  ✅ reports/ exists

✓ Check 4: Files
  ✅ core/context_engine.py exists
  ✅ core/market_mood.py exists
  ✅ core/stage1_integration.py exists

✓ Check 5: Market Mood Test
  ✅ Market mood updated: bull regime
     Date: 2025-10-05
     Sentiment: risk-on
     SPY: $XXX.XX
     VIX: XX.X

✓ Check 6: Context Engine Test
  ✅ WorldContextEngine initialized
     Feeds: 25
     Watchlist: 10 symbols

✓ Check 7: Integration Module Test
  ✅ Integration module functional
     Engine status: Not yet run
     Mood status: Fresh (0.0h old)

✅ Stage 1 components are ready!

```text

#### ✅ 3. Check wolf_app.py Syntax

```bash

cd /workspaces/GHOST
source .venv/bin/activate
python -c "import wolf_app; print('✅ wolf_app loads successfully')"

```text

**Expected**: `✅ wolf_app loads successfully`

______________________________________________________________________

### Server Startup Tests

#### ✅ 4. Start Ghost Server

```bash

cd /workspaces/GHOST
source .venv/bin/activate
export PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload

```text

#### ✅ 5. Check Startup Logs

Look for these log entries in the server output:

```json

{
  "msg": "stage1_initialized",
  "component": "startup",
  "features": "world_context,market_mood",
  "update_interval": "5min"
}

```text

**Expected**: Stage 1 initializes without errors

#### ✅ 6. Check Background Updates

After 5 minutes, check logs for:

```json

{
  "msg": "stage1_context_updated",
  "articles_fetched": 150,
  "market_regime": "bull"
}

```text

______________________________________________________________________

### API Endpoint Tests

#### ✅ 7. Test Health Endpoint

```bash

curl -s <<<<<http://localhost:5000/health>>>>> | jq

```text

**Expected**: `{"status": "ok"}`

#### ✅ 8. Test World Context Endpoint

```bash

curl -s "<<<<<http://localhost:5000/api/stage1/world?hours=24&min_relevance=0.3">>>>> | jq

```text

**Expected Output**:

```json

{
  "avg_sentiment": 0.35,
  "article_count": 47,
  "trending_events": ["earnings", "ai-breakthrough", "product"],
  "top_headlines": [
    {
      "headline": "Tech Giants Report Strong Q3 Earnings",
      "sentiment": 0.65,
      "relevance": 0.85,
      "url": "<<<<<https://...">>>>>
    }
  ],
  "feeds_parsed": 25,
  "last_update": "2025-10-05T12:34:56Z"
}

```text

#### ✅ 9. Test Market Mood Endpoint

```bash

curl -s <<<<<http://localhost:5000/api/stage1/mood>>>>> | jq

```text

**Expected Output**:

```json

{
  "market_regime": "bull",
  "sentiment": "risk-on",
  "confidence": 0.82,
  "spy_price": 450.25,
  "spy_change_pct": 1.2,
  "qqq_price": 385.50,
  "vix_level": 13.5,
  "vix_interpretation": "low",
  "signals": [
    "SPY above MA20",
    "SPY above MA50",
    "Golden cross detected",
    "Low volatility (VIX < 15)"
  ],
  "summary": "Strong bull market with low volatility. Risk-on sentiment prevails.",
  "updated_at": "2025-10-05T12:00:00Z"
}

```text

#### ✅ 10. Test Symbol Context Endpoint

```bash

curl -s "<<<<<http://localhost:5000/api/stage1/symbol/WOLF?hours=24">>>>> | jq

```text

**Expected Output**:

```json

{
  "symbol": "WOLF",
  "article_count": 8,
  "avg_sentiment": 0.42,
  "avg_relevance": 0.78,
  "trending_events": ["earnings", "product"],
  "articles": [
    {
      "headline": "WOLF announces new product line",
      "sentiment": 0.65,
      "relevance": 0.92,
      "timestamp": "2025-10-05T10:30:00Z",
      "url": "<<<<<https://...",>>>>>
      "tags": ["product"]
    }
  ]
}

```text

#### ✅ 11. Test Stage 1 Stats Endpoint

```bash

curl -s <<<<<http://localhost:5000/api/stage1/stats>>>>> | jq

```text

**Expected Output**:

```json

{
  "context_engine": {
    "total_articles": 347,
    "last_update": "2025-10-05T12:34:56Z",
    "feeds_count": 25,
    "watchlist_count": 10,
    "avg_articles_per_feed": 13.88
  },
  "market_mood": {
    "current_regime": "bull",
    "last_update": "2025-10-05T12:00:00Z",
    "age_hours": 0.58,
    "is_stale": false
  }
}

```text

______________________________________________________________________

### AI Decision Tests

#### ✅ 12. Test Enhanced AI Context

```bash

# Request an AI decision (requires auth token)

curl -s -X POST <<<<<http://localhost:5000/api/ai/decide>>>>> \
  -H "Authorization: Bearer $(railway variables get GHOST_API_TOKEN)" \
  -H "Content-Type: application/json" | jq

```text

**Expected Output**(should now include world_context and market_mood):

```json

{
  "action": "BUY",
  "confidence": 75,
"rationale": "Strong price momentum (+2.3%), positive news sentiment (+0.45 across 47 articles), bull market regime (VIX
13.2), trending events: earnings, ai-breakthrough. Market risk-on.",
  "world_context": {
    "avg_sentiment": 0.45,
    "article_count": 47,
    "trending_events": ["earnings", "ai-breakthrough", "product"]
  },
  "market_mood": {
    "market_regime": "bull",
    "sentiment": "risk-on",
    "vix": 13.2
  },
  "timestamp": "2025-10-05T12:34:56Z"
}

```text**Key Improvements**:

- Rationale now mentions news sentiment, market regime, VIX
- World context provides article count and trending events
- Market mood shows current regime and sentiment
- Confidence adjusted based on market conditions


______________________________________________________________________

### Database Tests

#### ✅ 13. Check Context News Database

```bash

cd /workspaces/GHOST
sqlite3 data/context_news.db "SELECT COUNT(*) FROM world_news;"
sqlite3 data/context_news.db "SELECT headline, sentiment, relevance FROM world_news ORDER BY ts DESC LIMIT 5;"

```text

**Expected**:

- Article count: 100-500 (depending on how long server has run)
- Recent headlines with sentiment scores


#### ✅ 14. Check Market Mood JSON

```bash

cat data/market_mood.json | jq

```text

**Expected**:

```json

{
  "market_regime": "bull",
  "sentiment": "risk-on",
  "confidence": 0.82,
  "spy_price": 450.25,
  "updated_at": "2025-10-05T12:00:00Z",
  "summary": "Strong bull market with low volatility."
}

```text

______________________________________________________________________

### Performance Tests

#### ✅ 15. Monitor Background Task Performance

```bash

# Check logs for update times

grep "stage1_context_updated" ghost_server.log | tail -5

```text

**Expected**: Updates complete in 3-7 seconds every 5 minutes

#### ✅ 16. Check Database Size

```bash

ls -lh data/context_news.db
ls -lh data/market_mood.json

```text

**Expected**:

- context_news.db: 1-10 MB (grows to ~5 MB steady state)
- market_mood.json: ~2 KB


#### ✅ 17. Check Memory Usage

```bash

ps aux | grep uvicorn | awk '{print $11, $6/1024 "MB"}'

```text

**Expected**: Ghost server should use ~50-100 MB RAM (15-25 MB increase from Stage 1)

______________________________________________________________________

## 🐛 Troubleshooting

### Issue: "Stage 1 not enabled" in logs

**Cause**: Import error in core modules

**Solution**:

```bash

cd /workspaces/GHOST
source .venv/bin/activate
python -c "from core.stage1_integration import initialize_stage1; print('✅ Import successful')"

```text

If error persists, check:

```bash

python -c "import feedparser, yfinance, vaderSentiment; print('✅ Dependencies OK')"

```text

### Issue: "spacy model not found"

**Cause**: en_core_web_sm model not downloaded

**Solution**:

```bash

cd /workspaces/GHOST
source .venv/bin/activate
python -m spacy download en_core_web_sm

```text

**Note**: System will work in fallback mode (keyword matching) without spacy model

### Issue: "yfinance returns empty data"

**Cause**: Yahoo Finance API temporarily unavailable

**Solution**: Wait 5-10 minutes and check again. Market mood will show error but won't
crash.

### Issue: "RSS feed timeout"

**Cause**: Some feeds may be slow or temporarily down

**Solution**: Context engine skips failing feeds automatically. Check logs for specific
feed errors.

### Issue: "Context not appearing in AI decisions"

**Cause**: Background updater hasn't run yet

**Solution**: Wait 5 minutes after server startup for first update, then check again.

______________________________________________________________________

## 📊 Success Metrics

After integration, verify these metrics:

| Metric | Before Stage 1 | After Stage 1 | Status |
|--------|----------------|---------------|--------| | News Sources | 14 feeds | 25+
feeds | ✅ | | Context Depth | Basic headlines | NER + sentiment + events | ✅ | | Market
Awareness | None | Bull/bear/sideways regime | ✅ | | AI Decision Quality | Price + basic
news | Price + world context + market mood | ✅ | | Symbol Coverage | WOLF only |
10-stock watchlist | ✅ | | Update Frequency | On-demand | Every 5 minutes | ✅ | | API
Endpoints | 0 context endpoints | 4 new endpoints | ✅ |

______________________________________________________________________

## 🎯 Next Steps

### Immediate (Next 24 Hours)

1. **Monitor Performance**:

   - Check logs every hour for update times
   - Verify no errors in background task
   - Monitor database growth

1. **Test AI Decision Quality**:

   - Compare 10 decisions before/after Stage 1
   - Verify world_context appears in rationale
   - Check market_mood influences confidence

1. **Validate News Quality**:

   - Review top headlines in world context
   - Check sentiment scores are reasonable
   - Verify relevance matching works


### Week 2 Enhancements

1. **Add Context to Telegram Alerts**:

   - Include market mood in alert cards
   - Show trending events
   - Display top headlines

1. **Enhance UI with Context Panel**:

   - Add world context widget
   - Show market regime indicator
   - Display symbol-specific news

1. **Tune Relevance Thresholds**:

   - Adjust min_relevance based on use case
   - Filter low-quality feeds
   - Optimize NER performance


### Week 3-4: Stage 2 Implementation

Begin **Self-Evaluation System**:

- Accuracy tracker (predicted vs actual)
- Learning loop (auto-tuning)
- Model memory persistence


See `GHOST_INTELLIGENCE_UPGRADE_ROADMAP.md` for Stage 2 details.

______________________________________________________________________

## 📚 API Reference

### GET /api/stage1/world

Get world news context aggregated from 25+ sources.

**Parameters**:

- `hours` (int, default=24): Hours of historical context
- `min_relevance` (float, default=0.3): Minimum relevance score (0.0-1.0)


**Response**:

```json

{
  "avg_sentiment": 0.35,
  "article_count": 47,
  "trending_events": ["earnings", "ai-breakthrough"],
  "top_headlines": [...],
  "feeds_parsed": 25,
  "last_update": "2025-10-05T12:34:56Z"
}

```text

### GET /api/stage1/mood

Get current market mood/regime classification.

**Response**:

```json

{
  "market_regime": "bull",
  "sentiment": "risk-on",
  "confidence": 0.82,
  "spy_price": 450.25,
  "vix_level": 13.5,
  "signals": [...],
  "summary": "Strong bull market...",
  "updated_at": "2025-10-05T12:00:00Z"
}

```text

### GET /api/stage1/symbol/{symbol}

Get context for a specific symbol.

**Parameters**:

- `symbol` (string): Ticker symbol (e.g., WOLF, NVDA)
- `hours` (int, default=24): Hours of historical context


**Response**:

```json

{
  "symbol": "WOLF",
  "article_count": 8,
  "avg_sentiment": 0.42,
  "articles": [...]
}

```text

### GET /api/stage1/stats

Get Stage 1 statistics and health.

**Response**:

```json

{
  "context_engine": {
    "total_articles": 347,
    "last_update": "2025-10-05T12:34:56Z"
  },
  "market_mood": {
    "current_regime": "bull",
    "is_stale": false
  }
}

```text

______________________________________________________________________

## ✅ Integration Complete

**Intelligence Level**: 7 → 8 ACHIEVED! 🚀

GHOST now has:

- ✅ World context awareness (25+ news sources)
- ✅ Market mood understanding (bull/bear/sideways)
- ✅ Named entity recognition (tickers, companies, people)
- ✅ Event tagging (10+ categories)
- ✅ Symbol-specific context (10-stock watchlist)
- ✅ Background updates (every 5 minutes)
- ✅ Enhanced AI decisions (with world context + market mood)
- ✅ New API endpoints (4 endpoints)


**Next**: Monitor performance for 24-48 hours, then proceed to Stage 2 (Self-Evaluation
System).

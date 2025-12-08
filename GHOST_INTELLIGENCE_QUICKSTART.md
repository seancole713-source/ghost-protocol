# 🚀 GHOST Intelligence Upgrade — Quick Start Guide

**Goal**: Implement Level 7 → 10 intelligence upgrade in 8-12 weeks

______________________________________________________________________

## 📦 INSTALL DEPENDENCIES (5 minutes)

```bash

# Navigate to GHOST directory

cd /workspaces/GHOST

# Install all free tools

pip install feedparser spacy vaderSentiment yfinance scikit-learn chromadb numpy pandas

# Download spaCy language model (40MB)

python -m spacy download en_core_web_sm

# Verify installation

python -c "import feedparser, spacy, vaderSentiment; print('✅ All dependencies installed')"

```text

______________________________________________________________________

## 🏗️ STAGE 1: CONTEXT AWARENESS (Week 1-2)

### Step 1: Create Directory Structure

```bash

mkdir -p core logs reports data

```text

### Step 2: Implement World Context Engine

Create `core/context_engine.py`:

```python

# core/context_engine.py

import feedparser
import spacy
import sqlite3
import time
import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from typing import List, Dict, Any

class WorldContextEngine:
    """Aggregates 25 news sources and extracts market context."""

    def __init__(self, feeds: List[str], db_path: str = "data/context_news.db"):
        self.feeds = feeds
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.nlp = spacy.load("en_core_web_sm")
        self.vader = SentimentIntensityAnalyzer()
        self._init_db()

    def _init_db(self):
        """Create news storage table."""
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS world_news (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts BIGINT NOT NULL,
                source TEXT,
                headline TEXT,
                url TEXT,
                summary TEXT,
                sentiment REAL,
                entities TEXT,
                relevance REAL,
                tags TEXT
            )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS idx_ts ON world_news(ts)")
        self.db.commit()

    def fetch_and_parse(self):
        """Fetch all RSS feeds and parse articles."""
        for feed_url in self.feeds:
            try:
                parsed = feedparser.parse(feed_url)
                for entry in parsed.entries[:20]:
                    self._process_article(entry, feed_url)
            except Exception as e:
                print(f"Feed error {feed_url}: {e}")

    def _process_article(self, entry, source):
        """Extract entities, sentiment, relevance from article."""
        headline = entry.get('title', '')
        summary = entry.get('summary', '')[:500]
        url = entry.get('link', '')

        # Skip duplicates

        cur = self.db.execute("SELECT COUNT(*) FROM world_news WHERE url=?", (url,))
        if cur.fetchone()[0] > 0:
            return

        # Named entity extraction

        doc = self.nlp(headline + " " + summary)
        entities = [ent.text for ent in doc.ents if ent.label_ in ('ORG', 'PERSON', 'GPE')]

        # Sentiment scoring (-1.0 to +1.0)

        sentiment = self.vader.polarity_scores(headline + " " + summary)['compound']

        # Relevance to watchlist

        watchlist = ['WOLF', 'NVDA', 'PLTR', 'TSLA', 'AMD', 'AAPL', 'MSFT', 'GOOGL', 'META', 'AMZN']
        text_upper = (headline + " " + summary).upper()
        matches = sum(1 for sym in watchlist if sym in text_upper)
        relevance = min(1.0, matches / 3.0)

        # Event tagging

        tags = self._extract_tags(headline + " " + summary)

        # Store

        self.db.execute("""
            INSERT INTO world_news (ts, source, headline, url, summary, sentiment, entities, relevance, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            int(time.time()),
            source,
            headline,
            url,
            summary,
            sentiment,
            json.dumps(entities),
            relevance,
            json.dumps(tags)
        ))
        self.db.commit()

    def _extract_tags(self, text: str) -> List[str]:
        """Extract event keywords."""
        keywords = {
            'bankruptcy': ['bankruptcy', 'chapter 11', 'restructuring'],
            'earnings': ['earnings', 'beat', 'miss', 'guidance'],
            'merger': ['merger', 'acquisition', 'm&a', 'takeover'],
            'product': ['launch', 'product', 'release', 'unveil'],
            'regulatory': ['fda', 'sec', 'investigation', 'lawsuit']
        }
        tags = []
        text_lower = text.lower()
        for tag, kws in keywords.items():
            if any(kw in text_lower for kw in kws):
                tags.append(tag)
        return tags

    def get_recent_context(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of last N hours."""
        cutoff = int(time.time()) - (hours * 3600)
        cur = self.db.execute("""
            SELECT AVG(sentiment) as avg_sentiment,
                   COUNT(*) as article_count,
                   COUNT(DISTINCT source) as source_count
            FROM world_news
            WHERE ts > ? AND relevance > 0.3
        """, (cutoff,))
        row = cur.fetchone()

        # Get top tags

        cur = self.db.execute("""
            SELECT tags, COUNT(*) as cnt
            FROM world_news
            WHERE ts > ? AND relevance > 0.3
            GROUP BY tags
            ORDER BY cnt DESC
            LIMIT 5
        """, (cutoff,))
        top_tags = [r[0] for r in cur.fetchall()]

        return {
            'avg_sentiment': round(row[0] or 0.0, 3),
            'article_count': row[1] or 0,
            'source_count': row[2] or 0,
            'trending_events': ', '.join(json.loads(t) for t in top_tags if t) if top_tags else []
        }

```text

### Step 3: Implement Market Mood Tracker

Create `core/market_mood.py`:

```python

# core/market_mood.py

import yfinance as yf
import json
import time
from typing import Dict, Any

def update_market_mood() -> Dict[str, Any]:
    """Update daily market mood snapshot."""
    try:
        spy = yf.Ticker("SPY")
        qqq = yf.Ticker("QQQ")
        vix = yf.Ticker("^VIX")

        spy_hist = spy.history(period="5d")
        qqq_hist = qqq.history(period="5d")
        vix_hist = vix.history(period="1d")

        if len(spy_hist) < 2 or len(vix_hist) < 1:
            return {'error': 'Insufficient market data'}

        spy_price = spy_hist['Close'].iloc[-1]
        spy_start = spy_hist['Close'].iloc[0]
        spy_change = ((spy_price / spy_start) - 1) * 100

        qqq_price = qqq_hist['Close'].iloc[-1]
        qqq_start = qqq_hist['Close'].iloc[0]
        qqq_change = ((qqq_price / qqq_start) - 1) * 100

        vix_current = vix_hist['Close'].iloc[-1]

        # Regime classification

        if vix_current < 15 and spy_change > 0:
            regime = "bull"
            sentiment = "risk-on"
        elif vix_current > 25 or spy_change < -2:
            regime = "bear"
            sentiment = "risk-off"
        else:
            regime = "sideways"
            sentiment = "neutral"

        mood = {
            "date": time.strftime("%Y-%m-%d"),
            "market_regime": regime,
            "spy_trend": f"{spy_change:+.1f}%",
            "qqq_trend": f"{qqq_change:+.1f}%",
            "vix": round(vix_current, 1),
            "sentiment": sentiment,
            "updated_at": int(time.time())
        }

        # Save to file

        with open("data/market_mood.json", "w") as f:
            json.dump(mood, f, indent=2)

        return mood

    except Exception as e:
        return {'error': str(e)}

```text

### Step 4: Test Implementation

```python

# test_context.py

from core.context_engine import WorldContextEngine
from core.market_mood import update_market_mood

# Test context engine

feeds = [
    "<<<<<https://www.reuters.com/business/rss",>>>>>
    "<<<<<https://www.marketwatch.com/rss/topstories">>>>>
]

print("Testing World Context Engine...")
engine = WorldContextEngine(feeds)
engine.fetch_and_parse()
context = engine.get_recent_context(hours=24)
print(f"✅ Context: {context}")

# Test market mood

print("\nTesting Market Mood Tracker...")
mood = update_market_mood()
print(f"✅ Mood: {mood}")

```text

### Step 5: Integrate to wolf_app.py

Add to `wolf_app.py`:

```python

# At top of wolf_app.py

from core.context_engine import WorldContextEngine
from core.market_mood import update_market_mood
import asyncio
import json

# Initialize (after NEWS_MANUAL_FEEDS defined)

feeds = [f.strip() for f in NEWS_MANUAL_FEEDS.split(',') if f.strip()]
context_engine = WorldContextEngine(feeds)

# Background job

async def context_updater():
    """Update context every 5 minutes."""
    while True:
        try:
            context_engine.fetch_and_parse()
            update_market_mood()
        except Exception as e:
            print(f"Context update error: {e}")
        await asyncio.sleep(300)  # 5 minutes

# Start on app startup

@APP.on_event("startup")
async def startup_context():
    asyncio.create_task(context_updater())

# Modify _build_ai_context()

def _build_ai_context() -> dict[str, Any]:
    ctx = {...}  # existing context

    # Add world context

    try:
        ctx['world_context'] = context_engine.get_recent_context(hours=24)
    except:
        ctx['world_context'] = {}

    # Add market mood

    try:
        with open('data/market_mood.json', 'r') as f:
            ctx['market_mood'] = json.load(f)
    except:
        ctx['market_mood'] = {}

    return ctx

```text

______________________________________________________________________

## 📊 STAGE 2: SELF-EVALUATION (Week 3-4)

### Implementation Files

- `core/accuracy_tracker.py` — Track forecast vs actual
- `core/learning_loop.py` — Auto-tune parameters
- `data/forecast_accuracy.db` — Persistence


**Full code**: See `GHOST_INTELLIGENCE_UPGRADE_ROADMAP.md` Stage 2

______________________________________________________________________

## 🧠 STAGE 3: STRATEGIC REASONING (Week 5-6)

### Implementation Files

- `core/reasoning_engine.py` — 4-layer thinking
- `logs/ghost_thoughts.log` — Reasoning traces
- `reports/ghost_journal_{date}.md` — Daily summaries


**Full code**: See `GHOST_INTELLIGENCE_UPGRADE_ROADMAP.md` Stage 3

______________________________________________________________________

## 🔮 STAGE 4: PATTERN RECOGNITION (Week 7-8)

### Implementation Files

- `core/pattern_recognition.py` — DBSCAN clustering
- Uses existing `core/ai_memory.py` for semantic search ✅


**Full code**: See `GHOST_INTELLIGENCE_UPGRADE_ROADMAP.md` Stage 4

______________________________________________________________________

## 🎯 STAGE 5: ADAPTIVE STRATEGIES (Week 9)

### Implementation Files

- `core/regime_detector.py` — Bull/bear/sideways detection
- Strategy switching per regime


**Full code**: See `GHOST_INTELLIGENCE_UPGRADE_ROADMAP.md` Stage 5

______________________________________________________________________

## 📈 STAGE 6: PORTFOLIO INTELLIGENCE (Week 10)

### Implementation Files

- `core/portfolio_analyzer.py` — Multi-stock correlation
- Risk-adjusted allocation


**Full code**: See `GHOST_INTELLIGENCE_UPGRADE_ROADMAP.md` Stage 6

______________________________________________________________________

## ✅ VALIDATION CHECKLIST

After each stage, verify:

### Stage 1 (Context)

- [ ] `data/context_news.db` has >100 articles
- [ ] `data/market_mood.json` updates daily
- [ ] `/api/news` shows enhanced context
- [ ] News sentiment scores include global context


### Stage 2 (Self-Evaluation)

- [ ] `data/forecast_accuracy.db` tracks predictions
- [ ] MAP computed weekly
- [ ] Auto-tuning triggers when MAP > 5%
- [ ] `data/model_memory.json` stores tuned params


### Stage 3 (Reasoning)

- [ ] `logs/ghost_thoughts.log` shows 4-layer reasoning
- [ ] Every decision has 2-line rationale
- [ ] `reports/ghost_journal_{date}.md` generates daily
- [ ] Telegram alerts include explanations


### Stage 4 (Patterns)

- [ ] Pattern recognition discovers 10+ clusters
- [ ] Semantic search finds similar scenarios
- [ ] Ghost pattern library persists
- [ ] Alerts fire on pattern matches


### Stage 5 (Adaptive)

- [ ] Regime detection runs daily
- [ ] Strategies switch per regime
- [ ] Bull/bear/sideways classification accurate


### Stage 6 (Portfolio)

- [ ] Correlation matrix computed
- [ ] High correlation pairs flagged (>0.8)
- [ ] Sector diversification checked
- [ ] Risk-adjusted allocation recommended


______________________________________________________________________

## 🚨 TROUBLESHOOTING

### Issue: "spaCy model not found"

**Solution**:

```bash

python -m spacy download en_core_web_sm
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('✅ Model loaded')"

```text

### Issue: "yfinance returns empty data"

**Solution**: Yahoo Finance rate limiting. Add delays:

```python

import time
time.sleep(1)  # Between requests

```text

### Issue: "ChromaDB initialization error"

**Solution**: Use SQLite-only mode:

```python

memory = AIMemory(vector_store="none")  # Disable ChromaDB

```text

### Issue: "RSS feed timeout"

**Solution**: Increase timeout, skip failing feeds:

```python

parsed = feedparser.parse(feed_url, timeout=10)

```text

______________________________________________________________________

## 📚 DOCUMENTATION

- **Full Roadmap**: `GHOST_INTELLIGENCE_UPGRADE_ROADMAP.md`
- **Existing AI Memory**: `core/ai_memory.py` (already complete ✅)
- **Evolution Plan**: `GHOST_EVOLUTION_PLAN.md`
- **News Feeds**: `NEWS_FEEDS_IMPLEMENTATION_COMPLETE.md`


______________________________________________________________________

## 🎓 LEARNING PATH

1. **Week 1**: Read roadmap, install dependencies, understand architecture
2. **Week 2**: Implement Stage 1 (Context), test with sample feeds
3. **Week 3-4**: Stage 2 (Self-evaluation), verify MAP tracking
4. **Week 5-6**: Stage 3 (Reasoning), check daily journals
5. **Week 7-8**: Stage 4 (Patterns), test similarity search
6. **Week 9**: Stage 5 (Adaptive), verify regime switching
7. **Week 10**: Stage 6 (Portfolio), validate correlations
8. **Week 11-12**: Integration testing, benchmarking, documentation


______________________________________________________________________

## 💡 QUICK WINS (First 2 Hours)

1. **Install dependencies**(10 min)


2.**Create directory structure**(2 min)
3.**Copy context_engine.py**(5 min)
4.**Copy market_mood.py**(5 min)
5.**Test both modules**(10 min)
6.**Integrate to wolf_app.py**(30 min)
7.**Restart server, verify working**(5 min)
8.**Check `data/context_news.db` populating**(10 min)**After 2 hours**: GHOST will have Level 8 context awareness! 🎉

______________________________________________________________________

## 🚀 NEXT STEPS

1. **Start with Stage 1**— Context awareness (easiest, highest impact)


2.**Test thoroughly**— Verify each component works before proceeding
3.**Iterate weekly**— Complete one stage per 1-2 weeks
4.**Document learnings**— Update daily journals
5.**Benchmark performance**— Track MAP, direction accuracy
6.**Celebrate milestones**— Each stage completion is a win!**Ready? Let's build Level 10 intelligence! 🧠⚡**

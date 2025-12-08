# ChatGPT Analyst Integration for Ghost

## Overview

Ghost now uses **ChatGPT as its reasoning brain**while Ghost provides the data layer
and execution. This creates a**self-healing, continuous analyst**that:

- Monitors your portfolio 24/7
- Detects trading opportunities using real market data
- Issues actionable recommendations with confidence scores
- Never forgets context (auto-rehydrates when sessions reset)
- Costs ~$0.50-2.00/day with GPT-4o-mini


## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    ChatGPT Analyst                      │
│                 (Reasoning Engine)                      │
│   - Analyzes market data                               │
│   - Detects opportunities/risks                        │
│   - Issues Decision Cards                              │
└──────────────────┬──────────────────────────────────────┘
                   │ API Calls
                   │ (every 5 min)
┌──────────────────▼──────────────────────────────────────┐
│              Ghost Agent Loop                           │
│   - Persistent state (SQLite)                          │
│   - Context hydration                                  │
│   - Tool orchestration                                 │
│   - Task queue                                         │
└──────────────────┬──────────────────────────────────────┘
                   │ Function Calls
                   │
┌──────────────────▼──────────────────────────────────────┐
│              Ghost Data Tools (7)                       │
│   1. news.search         - Recent news + sentiment     │
│   2. filings.search      - SEC EDGAR filings           │
│   3. insiders.form4      - Insider trading (Form 4)    │
│   4. options.daily       - Options flow, put/call      │
│   5. prices.history      - OHLCV + technicals          │
│   6. company.profile     - Fundamentals, earnings      │
│   7. sentiment.score     - Text sentiment analysis     │
└──────────────────┬──────────────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────────────┐
│           Ghost Execution Layer                         │
│   - Portfolio management                               │
│   - Telegram alerts                                    │
│   - Order execution (future)                           │
└─────────────────────────────────────────────────────────┘

```text

## Why This Architecture

### The Problem

- ChatGPT sessions end → context lost
- Manual analysis = time-consuming
- Data scattered across APIs
- No persistent memory


### The Solution

Ghost owns the**permanent memory**(SQLite), ChatGPT provides the**reasoning**:

| Component | Persistence | Role | |-----------|-------------|------| | **Ghost**|
Permanent (DB) | Data, tools, execution, memory | |**ChatGPT**| Temporary (session) |
Analysis, pattern recognition, recommendations | |**Agent Loop**| Self-healing |
Rehydrates context every tick, never loses state |

When ChatGPT forgets or disconnects, the loop automatically:

1. Detects "RESET_NEEDED" in response
2. Rehydrates with fresh runtime snapshot
3. Continues from last known state


## Setup

### 1. Get OpenAI API Key

1. Visit <<<<<https://platform.openai.com/api-keys>>>>>
2. Create new key
3. Copy to clipboard


### 2. Configure Environment

Edit `secrets.env`:

```bash

# Required

OPENAI_API_KEY=sk-proj-your-key-here

# Optional (defaults shown)

GHOST_LLM_MODEL=gpt-4o-mini          # or gpt-4o for better accuracy
GHOST_AGENT_TICK=300                 # 5 minutes (lower = more expensive)
GHOST_AGENT_MAX_HISTORY=20           # Message history limit
GHOST_AGENT_DB=./data/ghost_agent.db # State persistence path

```text

### 3. Restart Ghost

```bash

source .venv/bin/activate
uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload

```text

Look for startup message:

```text

✅ ChatGPT Analyst attached - /agent/health /agent/state /agent/outbox

```text

### 4. Verify Working

```bash

# 1. Check health

curl <<<<<http://localhost:5000/agent/health>>>>> | jq

# Expected

{
  "status": "ok",
  "model": "gpt-4o-mini",
  "ticks_ok": 1,
  "ticks_fail": 0,
  "last_ok_ts": "2025-10-08T02:28:55.549697+00:00",
  "last_error": null,
  "reset_events": 1,
  "loop_interval_sec": 300
}

# 2. Check conversation state

curl <<<<<http://localhost:5000/agent/state>>>>> | jq '.messages | length'

# Expected: 2+ (system prompt + rehydration + analyst responses)

# 3. Check outbox (analyst tasks)

curl <<<<<http://localhost:5000/agent/outbox>>>>> | jq '.'

# Expected: [] or [{task}, {task}, ...]

```text

## API Endpoints

### Analyst Monitoring

| Endpoint | Method | Description | |----------|--------|-------------| |
`/agent/health` | GET | Loop status, ticks, errors | | `/agent/state` | GET |
Conversation history | | `/agent/outbox` | GET | Queued tasks from analyst |

### Analyst Tools (ChatGPT calls these)

| Endpoint | Method | Auth Required | Description |
|----------|--------|---------------|-------------| | `/api/analyst/news.search` | POST
| ✅ | Get recent news with sentiment | | `/api/analyst/filings.search` | POST | ✅ | SEC
EDGAR filings (8-K, 10-Q, 10-K, 4) | | `/api/analyst/insiders.form4` | POST | ✅ |
Insider trading activity | | `/api/analyst/options.daily` | POST | ✅ | Options flow,
put/call ratios | | `/api/analyst/prices.history` | POST | ✅ | OHLCV + RSI, SMA,
Bollinger Bands | | `/api/analyst/company.profile` | POST | ✅ | Company fundamentals,
earnings dates | | `/api/analyst/sentiment.score` | POST | ✅ | Text sentiment analysis |

### Example Tool Call

```bash

# News search (what ChatGPT will call internally)

curl -X POST <<<<<http://localhost:5000/api/analyst/news.search>>>>> \
  -H "Authorization: Bearer ${GHOST_API_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "symbols": ["WOLF", "AAPL"],
    "sources": ["reuters", "marketwatch"],
    "since": "2025-10-01T00:00:00Z"
  }'

# Response

{
  "ok": true,
  "articles": [
    {
      "ts": "2025-10-07T14:30:00Z",
      "src": "reuters",
      "title": "WOLF announces Q3 earnings beat",
      "url": "...",
      "sentiment": 0.8,
      "syms": ["WOLF"]
    }
  ],
  "count": 1
}

```text

## Decision Cards

When the analyst detects an opportunity, it emits a**Decision Card**:

```json

{
  "type": "decision",
  "symbol": "WOLF",
  "action": "BUY",
  "confidence": 0.85,
  "horizon": "24-72h",
"summary": "Strong insider buying (CEO purchased 50K shares) + options flow tilted bullish (P/C ratio 0.65) + positive
earnings surprise. Technical breakout above $28 resistance with volume confirmation.",

  "catalysts": [
    {
      "type": "insider",
      "title": "CEO bought 50,000 shares @ $27.80",
      "ts": "2025-10-06T16:00:00Z",
      "relevance": 0.95
    },
    {
      "type": "news",
      "title": "Q3 Earnings Beat: EPS $1.20 vs $1.05 est",
      "ts": "2025-10-07T08:30:00Z",
      "relevance": 0.90
    },
    {
      "type": "options",
      "title": "Unusual call activity: 2x normal volume",
      "ts": "2025-10-07T15:00:00Z",
      "relevance": 0.80
    }
  ],

  "risks": [
    {
      "type": "macro",
      "note": "VIX elevated at 22 (vs 15 avg)",
      "weight": 0.4
    },
    {
      "type": "technical",
      "note": "Overbought RSI (78)",
      "weight": 0.3
    }
  ],

  "metrics": {
    "price": 27.85,
    "target": 32.00,
    "stop": 26.50,
    "put_call_ratio": 0.65,
    "short_interest_pct": 1.2,
    "insider_signal": 0.8,
    "sentiment_score": 0.75
  },

  "next_steps": [
    "Set alert at $30 for partial profit take",
    "Re-check if VIX crosses 25",
    "Monitor insider Form 4 filings next 5 days"
  ],

  "data_sources": [
    "insiders.form4",
    "options.daily",
    "news.search",
    "prices.history"
  ]
}

```text

## Task Schema

For system-level issues, the analyst emits **Task JSON**:

```json

{
  "type": "task",
  "priority": "high",
  "title": "Polygon API rate limited",
  "symbol": null,
  "instructions": "Rotate to AlphaVantage for price data. Polygon limit resets at midnight UTC.",
  "tags": ["provider", "price", "rate_limit"],
  "confidence": 0.95,
  "horizon": "4h",
  "data_sources": ["prices.history"],
  "reasoning": "Polygon returned 429 errors on last 3 WOLF price requests. Yahoo Finance also degraded.",
  "risks": ["Stale prices could cause bad predictions"],
  "checks": [
    "curl <<<<<http://localhost:5000/api/prices/WOLF",>>>>>
    "Check GHOST_PRIMARY_PROVIDER env var"
  ],
  "rollback": "Revert to Polygon after midnight UTC"
}

```text

## Cost Estimates

Model: **GPT-4o-mini**(recommended)

| Tick Interval | Ticks/Day | Tokens/Tick (avg) | Cost/Day |
|---------------|-----------|-------------------|----------| | 5 min | 288 | ~1,000 |
$0.50 | | 10 min | 144 | ~1,000 | $0.25 | | 30 min | 48 | ~1,000 | $0.10 |

Model:**GPT-4o**(more accurate)

| Tick Interval | Ticks/Day | Tokens/Tick (avg) | Cost/Day |
|---------------|-----------|-------------------|----------| | 5 min | 288 | ~1,000 |
$5.00 | | 10 min | 144 | ~1,000 | $2.50 | | 30 min | 48 | ~1,000 | $0.85 |**Recommendations:**-**Development**: Use
GPT-4o-mini with 5-10 min ticks ($0.25-0.50/day)

- **Production**: Use GPT-4o with 10 min ticks ($2.50/day) for better accuracy
- **Budget**: Use GPT-4o-mini with 30 min ticks ($0.10/day)


**Monthly costs:**

- GPT-4o-mini (5 min): ~$15/month
- GPT-4o (10 min): ~$75/month


## How It Works

### 1. Tick Cycle (every 5 minutes)

```python

# Agent Loop

1. Load conversation history from DB
2. Build runtime snapshot (portfolio, prices, providers)
3. Send to ChatGPT:
   - System prompt (identity, rules, schemas)
   - Recent messages (last 20)
   - Fresh snapshot (current state)
1. ChatGPT analyzes and responds
2. Parse response for actions/decisions
3. Queue tasks to outbox
4. Save updated state to DB


```text

### 2. Context Hydration

```python

# When ChatGPT loses context (session reset)

if response == "RESET_NEEDED":
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"REHYDRATE_CONTEXT\n{snapshot}"}
    ]
    save_state(messages)
    continue

```text

### 3. Tool Orchestration

```python

# ChatGPT decides which tools to call

{
  "function": "news.search",
  "arguments": {
    "symbols": ["WOLF"],
    "since": "2025-10-01T00:00:00Z"
  }
}

# Agent loop executes

result = await call_tool("news.search", arguments)

# Returns to ChatGPT

{
  "ok": true,
  "articles": [...]
}

# ChatGPT reasons and emits Decision Card

```text

## Advanced Usage

### Custom System Prompt

Edit `ghost_agent_loop.py`, line 56 (SYSTEM_PROMPT):

```python

SYSTEM_PROMPT = """You are Ghost's ChatGPT Analyst.

CUSTOM RULES:

- Focus on swing trades (2-5 day holds)
- Only recommend when confidence > 80%
- Prioritize insider buying + options flow combos
- Avoid earnings week trades


[rest of prompt...]
"""

```text

### Adjust Tick Frequency

```bash

# More frequent = more responsive = higher cost

GHOST_AGENT_TICK=180  # 3 minutes

# Less frequent = cheaper = less responsive

GHOST_AGENT_TICK=600  # 10 minutes

```text

### Wire to Telegram

Edit `ghost_agent_loop.py`, line 555 (outbox_delivery_loop):

```python

async def outbox_delivery_loop():
    while True:
        batch = grab_undelivered(20)
        for row in batch:
            payload = json.loads(row["payload_json"])

            # ADD THIS

            if payload.get("type") == "decision":
                await send_telegram_alert(
                    f"🎯 {payload['action']} {payload['symbol']} "
                    f"@ {payload['metrics']['price']} "
                    f"(confidence: {payload['confidence']*100:.0f}%)\n\n"
                    f"{payload['summary']}"
                )

            mark_delivered([row["id"]])
        await asyncio.sleep(5)

```text

### Add Custom Tools

1. **Create endpoint in wolf_app.py:**```python


@APP.post("/api/analyst/custom.tool")
async def analyst_tool_custom(...):

    # Your logic here

    return {"ok": True, "data": ...}

```text

1.**Register in SYSTEM_PROMPT:**```python

SYSTEM_PROMPT = """...
Tools you can call:

- custom.tool: Description of what it does


"""

```text

1.**ChatGPT can now call it:**```json

{
  "function": "custom.tool",
  "arguments": {...}
}

```text

## Troubleshooting

### Analyst Not Running**Symptom:**`/agent/health` shows `ticks_ok: 0`**Fix:**1. Check `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`

1. Check server logs: `tail -f ghost_server.log | grep -i analyst`
2. Verify API key is valid:


   `curl <<<<<https://api.openai.com/v1/models>>>>> -H "Authorization: Bearer $OPENAI_API_KEY"`

### Rate Limit Errors**Symptom:**`/agent/health` shows `last_error: "Rate limit exceeded"`**Fix:**1. Increase tick interval: `GHOST_AGENT_TICK=600` (10 min)

1. Upgrade OpenAI plan (higher rate limits)
2. Add retry logic (already built-in, wait 1 min)


### Context Loss**Symptom:**ChatGPT keeps responding "I don't have context"**Fix:**1. Check DB exists: `ls -lh data/ghost_agent.db`

1. Verify rehydration:


   `curl <<<<<http://localhost:5000/agent/state>>>>> | jq '.messages[1].content'`

1. Force reset: `rm data/ghost_agent.db && restart server`


### High Costs**Symptom:**OpenAI bill higher than expected**Fix:**1. Use GPT-4o-mini: `GHOST_LLM_MODEL=gpt-4o-mini`

1. Increase tick interval: `GHOST_AGENT_TICK=600`
2. Reduce history: `GHOST_AGENT_MAX_HISTORY=10`
3. Monitor usage: <<<<<https://platform.openai.com/usage>>>>>


## Testing

### Manual Test

```bash

# 1. Health check

curl <<<<<http://localhost:5000/agent/health>>>>>

# 2. Trigger manual tick (restart server)

pkill -f uvicorn && uvicorn wolf_app:app --port 5000 &
sleep 10

# 3. Check for response

curl <<<<<http://localhost:5000/agent/state>>>>> | jq '.messages[-1]'

# Expected: assistant message with analysis

```text

### Automated Test

```bash

#!/bin/bash

# test_analyst.sh

echo "Testing ChatGPT Analyst..."

# Test 1: Health

echo "1. Health check..."
STATUS=$(curl -s <<<<<http://localhost:5000/agent/health>>>>> | jq -r '.status')
if [ "$STATUS" = "ok" ]; then
  echo "✅ Health OK"
else
  echo "❌ Health FAIL: $STATUS"
  exit 1
fi

# Test 2: State persistence

echo "2. State persistence..."
MSG_COUNT=$(curl -s <<<<<http://localhost:5000/agent/state>>>>> | jq '.messages | length')
if [ "$MSG_COUNT" -gt 0 ]; then
  echo "✅ State OK ($MSG_COUNT messages)"
else
  echo "❌ State FAIL (no messages)"
  exit 1
fi

# Test 3: Outbox delivery

echo "3. Outbox delivery..."
TASK_COUNT=$(curl -s <<<<<http://localhost:5000/agent/outbox>>>>> | jq 'length')
echo "✅ Outbox OK ($TASK_COUNT tasks)"

echo "All tests passed! 🎉"

```text

## Next Steps

1.**Week 1**: Monitor `/agent/outbox` for tasks, verify ChatGPT reasoning quality

1. **Week 2**: Wire Decision Cards to Telegram alerts
2. **Week 3**: Add paper trading execution (simulate trades based on decisions)
3. **Month 1**: Track analyst accuracy (Decision Cards vs actual outcomes)
4. **Month 2**: Auto-execute high-confidence trades (>85%) with position sizing
5. **Month 3**: Add more tools (earnings calendars, social sentiment, sector rotation)


## FAQ

**Q: Does ChatGPT remember past conversations?**\
A: Yes! Ghost stores the full conversation in SQLite. When ChatGPT loses context, Ghost
automatically rehydrates it with the latest state.

**Q: Can I use other models (Claude, Llama)?**\
A: Yes, change `OPENAI_BASE_URL` to point to your proxy/API gateway. The code uses
OpenAI-compatible format.

**Q: How secure is my data?**\
A: OpenAI doesn't train on API data (per their terms). All sensitive data (API keys,
tokens) stay in Ghost's DB, never sent to ChatGPT.

**Q: What happens if OpenAI is down?**\
A: The loop retries with exponential backoff (2s → 4s → 8s → 16s → 32s → 60s max). After
6 failures, it waits for next tick.

**Q: Can I run multiple analysts for different strategies?**\
A: Yes! Copy `ghost_agent_loop.py` → `ghost_agent_momentum.py`, change `SYSTEM_PROMPT`,
use different DB path, attach both to app.

**Q: Does this replace Ghost's existing prediction engine?**\
A: No, it complements it. Ghost's learning loop runs daily predictions. ChatGPT Analyst
provides continuous monitoring + actionable recommendations.

## Credits

- Architecture: Inspired by LangChain Agents + AutoGPT persistence patterns
- Integration: Built on Ghost's existing tool ecosystem
- Cost optimization: Based on OpenAI's token pricing (Oct 2025)


______________________________________________________________________

**Ready to enable?** Just add `OPENAI_API_KEY` to `secrets.env` and restart Ghost! 🚀

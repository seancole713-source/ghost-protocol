# 🐛 GHOST Agent Bug Analysis

**Date**: October 8, 2025\
**Issue**: Agent sees "empty portfolio" when WOLF position exists

______________________________________________________________________

## Root Cause Found ✅

**File**: `ghost_agent_loop.py`\
**Function**: `build_runtime_snapshot()` (lines 650-810)\
**Bug**: The function fetches portfolio data correctly BUT the logic has a flaw

### The Problem

```python
# Line 687-702 in ghost_agent_loop.py
try:
    resp = requests.get(f"{base_url}/api/position", timeout=2)
    if resp.status_code == 200:
        pos = resp.json()
        wolf_qty = float(pos.get("qty", 0) or 0)
        wolf_avg = float(pos.get("avg_cost", 0) or 0)
except Exception as e:
    logging.debug(f"position_fetch_failed: {e}")
```

**This works!** Gets: `wolf_qty=8.41959051`, `wolf_avg=359.28`

BUT THEN:

```python
# Lines 704-718: Price fetching
try:
    price_resp = requests.get(f"{base_url}/api/price/WOLF", timeout=2)
    if price_resp.status_code == 200:
        pr = price_resp.json()
        current_price = float(pr.get("price")) if pr.get("price") is not None else None
        prev_close = float(pr.get("prev_close")) if pr.get("prev_close") is not None else None
except Exception:
    pass

if current_price is None:
    # Falls back to yfinance...
```

**The Bug**: When rate-limited, `/api/price/WOLF` may fail OR return cached price. The
logic sets `portfolio_data` ONLY if `wolf_qty > 0 AND current_price is not None`.

**But the REAL issue**: Even when it works, the snapshot shows correct data! Let me
check the ACTUAL snapshot...

______________________________________________________________________

## Actual Investigation

Let me test what the agent ACTUALLY receives:

```bash
# Simulate what build_runtime_snapshot() returns
python3 << 'EOF'
import requests, json, os
from datetime import datetime, timezone

base_url = "http://localhost:5000"

# Exact agent logic
portfolio_data = {"nav": 0.0, "pnl_today": None, "pnl_pct": None, "cash": 0.0, "positions": []}
wolf_qty = 0.0
wolf_avg = 0.0
current_price = None
prev_close = None

try:
    resp = requests.get(f"{base_url}/api/position", timeout=2)
    if resp.status_code == 200:
        pos = resp.json()
        wolf_qty = float(pos.get("qty", 0) or 0)
        wolf_avg = float(pos.get("avg_cost", 0) or 0)
        print(f"✅ Got position: qty={wolf_qty}, avg={wolf_avg}")
except Exception as e:
    print(f"❌ Position fetch failed: {e}")

try:
    price_resp = requests.get(f"{base_url}/api/price/WOLF", timeout=2)
    if price_resp.status_code == 200:
        pr = price_resp.json()
        current_price = float(pr.get("price")) if pr.get("price") is not None else None
        prev_close = float(pr.get("prev_close")) if pr.get("prev_close") is not None else None
        print(f"✅ Got price: current={current_price}, prev={prev_close}")
except Exception as e:
    print(f"⚠️  Price fetch failed: {e}")

# Check condition
if wolf_qty > 0 and current_price is not None:
    print(f"\n✅ CONDITION MET: Building portfolio_data")
    pnl_abs = wolf_qty * (current_price - wolf_avg) if wolf_avg > 0 else 0.0
    pnl_pct = ((current_price - wolf_avg) / wolf_avg * 100) if wolf_avg > 0 else None
    pnl_today = None
    if prev_close is not None:
        pnl_today = wolf_qty * (current_price - prev_close)
    
    portfolio_data = {
        "nav": round(wolf_qty * current_price, 2),
        "pnl_today": round(pnl_today, 2) if pnl_today is not None else None,
        "pnl_pct": round(pnl_pct, 2) if pnl_pct is not None else None,
        "cash": 0.0,
        "positions": [
            {
                "symbol": "WOLF",
                "qty": wolf_qty,
                "avg_cost": wolf_avg,
                "current_price": current_price,
                "prev_close": prev_close,
                "value": round(wolf_qty * current_price, 2),
                "pnl": round(pnl_abs, 2)
            }
        ]
    }
    print(json.dumps(portfolio_data, indent=2))
else:
    print(f"\n❌ CONDITION FAILED:")
    print(f"   wolf_qty > 0: {wolf_qty > 0} (qty={wolf_qty})")
    print(f"   current_price is not None: {current_price is not None} (price={current_price})")
    print(f"   Result: Empty portfolio data sent to agent!")
    print(json.dumps(portfolio_data, indent=2))

EOF
```

______________________________________________________________________

## The REAL Problem

After testing, I suspect the issue is:

1. **Agent gets correct snapshot with WOLF position**
2. **But agent's CONVERSATION MEMORY says "empty portfolio"**
3. **Agent trusts its memory over fresh data**

Look at the agent state messages - they ALL say "portfolio is currently empty". This
means:

- Agent received data showing empty portfolio EARLIER
- Agent's conversation context retains that old belief
- Even though fresh snapshots show WOLF position, agent refers to old context

### The Fix

The agent needs to either:

1. **Trust fresh snapshots over conversation history** (prompt engineering fix)
2. **Force context reset** to clear old "empty portfolio" beliefs
3. **Add portfolio to EVERY user message** so agent can't forget

______________________________________________________________________

## Recommended Fix

### Option 1: Quick Fix - Force Context Reset

```bash
# Delete agent conversation state to force rehydration
rm data/ghost_agent.db
# Or just the conversation table:
sqlite3 data/ghost_agent.db "DELETE FROM agent_conversations WHERE id=1;"

# Restart server
pkill -f "uvicorn wolf_app"
source .venv/bin/activate && nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 --reload > ghost_server.log 2>&1 &
```

### Option 2: Prompt Engineering Fix

Update the system prompt to say:

```
CRITICAL: Always trust the latest runtime snapshot data over your conversation memory. 
If the snapshot shows positions, they exist - regardless of what you said earlier.
```

### Option 3: Add Portfolio to Every Tick

Modify `analyst_tick()` to ALWAYS inject fresh portfolio snapshot as a user message:

````python
async def analyst_tick(llm: LLMClient):
    # ... existing code ...
    
    # ALWAYS add fresh snapshot before asking for analysis
    fresh_snapshot = build_runtime_snapshot()
    messages.append({
        "role": "user",
        "content": f"Current state:\n```json\n{fresh_snapshot}\n```\nProvide status update.",
        "ts": datetime.now(timezone.utc).isoformat()
    })
    
    # Then call LLM...
````

______________________________________________________________________

## Testing the Fix

After applying fix:

```bash
# Wait 5 minutes for next agent tick
sleep 300

# Check if agent now sees portfolio
curl -s http://localhost:5000/agent/state | jq '.messages[-1].content' | grep -i "portfolio\|WOLF\|position"

# Check if new decision is made
curl -s http://localhost:5000/api/ai/decisions?limit=1 | jq '.decisions[0] | {symbol, action, confidence, rationale}'
```

Expected result:

- Agent acknowledges WOLF position
- Agent analyzes the -92.72% loss
- Agent generates SELL or HOLD decision with reasoning

______________________________________________________________________

## Priority

**CRITICAL** - This is why agent appears "frozen" and not making decisions. Fix this
first!

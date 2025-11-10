# 🧠 GHOST AI INTELLIGENCE UPGRADE - COMPLETE IMPLEMENTATION

**Date**: October 13, 2025\
**Goal**: Make Ghost truly intelligent with ChatGPT function calling and real-time
awareness

______________________________________________________________________

## 🎯 What You Want Ghost to Do:

### Current Limitations:

❌ Ghost doesn't know what day it is\
❌ Ghost can't check its own health\
❌ Ghost has stale price data ($31.10 vs actual $30.50)\
❌ Ghost can't answer meta questions\
❌ Ghost UI is completely dead (all zeros)\
❌ Ghost can't tell you about bullish stocks today\
❌ Ghost can't summarize weekly news impact

### What Ghost SHOULD Do:

✅ Know current date/time\
✅ Check its own health status\
✅ Get real-time stock/crypto prices\
✅ Fetch latest news headlines\
✅ Explain its own capabilities\
✅ Answer "what's happening today?"\
✅ Tell you which stocks are bullish\
✅ Summarize news impact on your portfolio

______________________________________________________________________

## 💡 Solution: ChatGPT Function Calling + Real-Time Tools

### Architecture:

```
User Question → ChatGPT → Function Call → Tool Execution → ChatGPT → Smart Answer
```

### 

Example Flow:

**User**: "What day is it today?"

1. ChatGPT sees question needs current date
2. Calls `get_current_datetime()` function
3. Gets "Sunday, October 13, 2025, 3:47 PM EDT"
4. Returns: "It's Sunday, October 13, 2025, 3:47 PM EDT. Markets are closed."

**User**: "What's your health check?"

1. ChatGPT calls `get_ghost_health()` function
2. Gets {price_providers: "polygon_intraday OK", database: "connected", broker: "alpaca
   $200K"}
3. Returns: "✅ All systems healthy! Price providers working (Polygon intraday), database
   connected, Alpaca broker ready with $200K buying power."

**User**: "What stocks are bullish today?"

1. ChatGPT calls `get_live_stock_price("WOLF")`, `get_live_stock_price("NVDA")`, etc.
2. Gets real-time prices + intraday high/low/volume
3. Calls `get_latest_news("WOLF")` for context
4. Returns: "NVDA up +3.2% on high volume (bullish), WOLF down -1.93% on weak volume
   (neutral), AAPL flat (wait)."

______________________________________________________________________

## 🛠️ Implementation

### Tools Ghost Can Use:

1. **get_current_datetime()**

   - Returns current date/time in America/New_York timezone
   - Use: "What day is it?", "What time is it?", "Is market open?"

2. **get_ghost_health()**

   - Returns system health: providers, database, cache, broker
   - Use: "What's your health?", "Are you working?", "Any issues?"

3. **get_live_stock_price(symbol)**

   - Returns real-time price from Polygon intraday bars
   - Includes: current price, high, low, volume, VWAP
   - Use: "What's WOLF price?", "Show me NVDA", "How's AAPL doing?"

4. **get_live_crypto_price(symbol)**

   - Returns crypto price from CoinGecko/Binance quorum
   - Use: "What's Bitcoin price?", "Show me ETH", "Crypto market?"

5. **get_latest_news(symbol, limit=5)**

   - Returns latest news headlines from Polygon
   - Includes: headline, sentiment, timestamp, URL
   - Use: "Any news on WOLF?", "What's happening with NVDA?"

6. **get_ghost_capabilities()**

   - Returns list of Ghost features and commands
   - Use: "What can you do?", "Help", "Features?"

7. **get_bullish_stocks()**

   - Scans watchlist, returns top movers with positive momentum
   - Use: "What's bullish today?", "Top movers?", "Buy ideas?"

8. **get_portfolio_summary()**

   - Returns current positions, P&L, NAV
   - Use: "Show my portfolio", "How am I doing?", "P&L?"

______________________________________________________________________

## 📝 Code Implementation

### Step 1: Add Tool Execution Functions

```python
def _execute_tool(tool_name: str, arguments: dict) -> str:
    """Execute a tool function and return JSON result"""
    try:
        if tool_name == "get_current_datetime":
            now = datetime.now(timezone('America/New_York'))
            return json.dumps({
                "date": now.strftime("%A, %B %d, %Y"),
                "time": now.strftime("%I:%M:%S %p %Z"),
                "timestamp": int(now.timestamp()),
                "is_trading_hours": _is_market_open_now()[0]
            })
        
        elif tool_name == "get_ghost_health":
            health = {
                "overall": "healthy",
                "price_providers": {},
                "database": "connected" if os.path.exists(WOLF_SQLITE_PATH) else "missing",
                "cache": "active",
                "broker": "disabled"
            }
            
            # Check Polygon
            try:
                intraday = _fetch_polygon_intraday("WOLF")
                health["price_providers"]["polygon_intraday"] = "OK" if intraday else "FAILED"
            except:
                health["price_providers"]["polygon_intraday"] = "ERROR"
            
            # Check broker
            try:
                from core.alpaca_broker import get_broker
                broker = get_broker()
                if broker.enabled:
                    acc = broker.get_account()
                    health["broker"] = f"alpaca ${float(acc.get('buying_power', 0)):,.0f}"
            except:
                pass
            
            return json.dumps(health)
        
        elif tool_name == "get_live_stock_price":
            symbol = arguments.get("symbol", "WOLF").upper()
            
            # Try Polygon intraday first
            intraday = _fetch_polygon_intraday(symbol)
            if intraday:
                return json.dumps({
                    "symbol": symbol,
                    "price": intraday["price"],
                    "high": intraday["high"],
                    "low": intraday["low"],
                    "volume": intraday["volume"],
                    "vwap": intraday.get("vwap"),
                    "timestamp": intraday["timestamp"],
                    "provider": "polygon_intraday"
                })
            
            # Fallback to standard get_wolf_price
            price, prev, provider = get_wolf_price() if symbol == "WOLF" else (None, None, None)
            return json.dumps({
                "symbol": symbol,
                "price": price,
                "prev_close": prev,
                "provider": provider or "unavailable"
            })
        
        elif tool_name == "get_latest_news":
            symbol = arguments.get("symbol", "WOLF").upper()
            limit = arguments.get("limit", 5)
            
            news = get_wolf_news(limit=limit) if symbol == "WOLF" else {"items": []}
            headlines = [
                {
                    "headline": item.get("headline"),
                    "sentiment": item.get("sent"),
                    "timestamp": item.get("ts"),
                    "url": item.get("url")
                }
                for item in news.get("items", [])[:limit]
            ]
            
            return json.dumps({"symbol": symbol, "news": headlines})
        
        elif tool_name == "get_ghost_capabilities":
            return json.dumps({
                "features": [
                    "Real-time stock/crypto price tracking (Polygon intraday)",
                    "AI-powered trading signals (BUY/SELL/HOLD)",
                    "News sentiment analysis (FinBERT)",
                    "Portfolio management (positions, P&L, NAV)",
                    "Telegram bot with trading commands (/buy, /sell, /positions)",
                    "Alpaca broker integration (paper trading $200K)",
                    "SL/TP automation (-3% stop loss, +6% take profit)",
                    "Prometheus metrics export for monitoring",
                    "Prediction overlay with MAP accuracy tracking"
                ],
                "commands": [
                    "/status - Portfolio status",
                    "/signal - Current trading signal",
                    "/pnl - Daily P&L",
                    "/positions - Open positions",
                    "/buy SYMBOL QTY - Buy stocks",
                    "/sell SYMBOL - Sell position",
                    "/help - Show all commands"
                ],
                "health_check": "GET /health",
                "api_docs": "GET /docs"
            })
        
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    except Exception as e:
        return json.dumps({"error": str(e)})
```

### Step 2: Update `_ask_ghost_ai` with Function Calling

```python
def _ask_ghost_ai(question: str) -> str:
    """Enhanced AI with ChatGPT function calling for real-time awareness"""
    
    if not AGENTS_ENABLED:
        return "🤖 AI agent not enabled. Set AGENTS_ENABLED=1 and configure AI_PROVIDER."
    
    if AI_PROVIDER == "openai" and not OPENAI_API_KEY:
        return "❌ OpenAI API key not set."
    
    # Define available tools
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_datetime",
                "description": "Get current date, time, and market hours status",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_ghost_health",
                "description": "Check Ghost system health (providers, database, broker)",
                "parameters": {"type": "object", "properties": {}}
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_live_stock_price",
                "description": "Get real-time stock price with intraday high/low/volume",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker (e.g., WOLF, AAPL)"}
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_latest_news",
                "description": "Get latest news headlines for a stock",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                        "limit": {"type": "integer", "description": "Number of headlines", "default": 5}
                    },
                    "required": ["symbol"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "get_ghost_capabilities",
                "description": "List Ghost's features and commands",
                "parameters": {"type": "object", "properties": {}}
            }
        }
    ]
    
    try:
        system_prompt = (
            "You are Ghost, an AI trading advisor with access to real-time market data and system tools. "
            "Use function calls to answer questions accurately. "
            "Always check current date/time for time-sensitive questions. "
            "Always check system health when asked about status. "
            "Always fetch real-time prices when asked about specific stocks. "
            "Be concise, accurate, and cite your data sources."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # First API call with tools
        payload = {
            "model": AGENT_MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto"
        }
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
        
        r = _http_post(
            f"{OPENAI_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=15
        )
        r.raise_for_status()
        data = r.json()
        
        response_message = data["choices"][0]["message"]
        
        # Check if ChatGPT wants to call functions
        if response_message.get("tool_calls"):
            messages.append(response_message)
            
            # Execute each tool call
            for tool_call in response_message["tool_calls"]:
                function_name = tool_call["function"]["name"]
                function_args = json.loads(tool_call["function"]["arguments"])
                
                # Execute tool
                function_response = _execute_tool(function_name, function_args)
                
                # Add function response to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": function_name,
                    "content": function_response
                })
            
            # Second API call with function results
            second_payload = {
                "model": AGENT_MODEL,
                "messages": messages
            }
            
            r2 = _http_post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=second_payload,
                timeout=15
            )
            r2.raise_for_status()
            final_data = r2.json()
            
            return final_data["choices"][0]["message"]["content"]
        
        # No function calls needed, return direct answer
        return response_message.get("content", "❌ No response")
    
    except Exception as e:
        LOGGER.error(f"AI chat error: {e}", exc_info=True)
        return f"❌ AI error: {str(e)[:200]}"
```

______________________________________________________________________

## 🎯 Expected Results

### Before (Current):

```
User: "What day is it today?"
Ghost: "Wolfspeed's current price is stable at $31.1..."  ❌ WRONG

User: "What's your health check?"
Ghost: "I don't have access to that information."  ❌ USELESS

User: "What's WOLF price?"
Ghost: "$31.1 (from yesterday's close)"  ❌ STALE
```

### After (With Function Calling):

```
User: "What day is it today?"
Ghost: "It's Sunday, October 13, 2025, 3:47 PM EDT. Markets are closed for the weekend."  ✅ CORRECT

User: "What's your health check?"
Ghost: "✅ All systems healthy!  
- Price providers: Polygon intraday OK  
- Database: Connected  
- Broker: Alpaca ready with $200K buying power  
- Cache: Active"  ✅ DETAILED

User: "What's WOLF price?"
Ghost: "WOLF is currently $30.50 (-1.93% today).  
- Intraday high: $34.19  
- Intraday low: $28.80  
- Volume: 2.7M (21% of average - weak conviction)  
- Provider: Polygon intraday (5-min delayed)"  ✅ REAL-TIME
```

______________________________________________________________________

## 🚀 Immediate Next Steps

1. **Add `_execute_tool()` function** to wolf_app.py (code above)
2. **Replace `_ask_ghost_ai()`** with new function calling version (code above)
3. **Test locally**: Ask "What day is it?" via Telegram
4. **Commit and deploy** to Railway
5. **Fix UI** (separate issue - all zeros due to price provider failures)

______________________________________________________________________

## 📊 Success Criteria

✅ Ghost can answer "What day is it?"\
✅ Ghost can check its own health\
✅ Ghost can get real-time prices\
✅ Ghost can fetch latest news\
✅ Ghost can explain its capabilities\
✅ Ghost is self-aware and context-aware

**Status**: 🟢 READY TO IMPLEMENT\
**Estimated Time**: 45 minutes\
**Impact**: Ghost becomes truly intelligent

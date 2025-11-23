# 🚀 Quick Start: Enable Ghost AI Chat

## What Happened

You texted Ghost via Telegram and got:

```
Commands: /status, /signal, /pnl
```

This means:

1. ✅ Telegram webhook **IS working**
2. ❌ AI agent **is NOT enabled yet**

The code is ready, but you need to enable the AI provider.

______________________________________________________________________

## Enable AI Chat (2 Steps)

### Step 1: Set Your OpenAI Key

```bash
export OPENAI_API_KEY="$(railway variables get OPENAI_API_KEY)"
```

### Step 2: Run This Script

```bash
./enable_ghost_chat.sh openai
```

That's it! 🎉

______________________________________________________________________

## Or Do It Manually

```bash
# 1. Export environment variables
export AGENTS_ENABLED=1
export AI_PROVIDER=openai
export OPENAI_API_KEY="$(railway variables get OPENAI_API_KEY)"
export AGENT_MODEL="gpt-4o-mini"

# 2. Restart Ghost with env vars
pkill -f "uvicorn.*wolf_app"
cd /workspaces/GHOST
source .venv/bin/activate

AGENTS_ENABLED=1 \
AI_PROVIDER=openai \
AGENT_MODEL=gpt-4o-mini \
OPENAI_API_KEY="$OPENAI_API_KEY" \
nohup python -m uvicorn wolf_app:APP --host 0.0.0.0 --port 5000 --reload > ghost_server.log 2>&1 &
```

______________________________________________________________________

## Test It

Once enabled, text your Ghost bot:

```
What would a Bitcoin drop do to WOLF?
```

Ghost will respond:

```
🤔 Thinking...

🤖 Ghost:

A Bitcoin drop would have minimal direct impact on Wolfspeed (WOLF) 
as they operate in different sectors. However, broader market sentiment 
shifts from crypto volatility could pressure tech stocks including WOLF. 
Monitor NASDAQ correlation more than Bitcoin itself. [continues with analysis]
```

______________________________________________________________________

## Current Status Check

**Telegram Webhook:** ✅ Working (you got the fallback response) **AI Agent:** ❌ Not
enabled yet (needs `AGENTS_ENABLED=1` + API key) **Code:** ✅ Updated and deployed
(server restarted with `--reload`)

______________________________________________________________________

## What Ghost Can Answer

### Market Questions

- "What would a Bitcoin drop do to WOLF?"
- "How does Fed policy affect semiconductor stocks?"
- "Is WOLF correlated with NVDA?"

### Position Questions

- "Should I buy more WOLF today?"
- "When should I take profits?"
- "What's my risk exposure?"

### News Questions

- "How is the latest news affecting WOLF?"
- "What does negative sentiment mean for my position?"
- "Should I hold through earnings?"

### Technical Questions

- "Is WOLF oversold right now?"
- "What's the current trend?"
- "Should I buy this dip?"

______________________________________________________________________

## Cost

- **GPT-4o-mini**: ~$0.0001 per question (practically free!)
- **GPT-4**: ~$0.01 per question
- **Ollama**: FREE (but needs local GPU)

______________________________________________________________________

## Troubleshooting

### "AI agent not enabled"

**Solution:** Run `./enable_ghost_chat.sh openai`

### "Commands: /status, /signal, /pnl"

**Solution:** AI not enabled yet (same as above)

### "OpenAI API error"

**Solution:** Check your API key is valid and has credits

### Telegram not receiving messages

**Solution:** Check webhook is set:

```bash
curl "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getWebhookInfo"
```

______________________________________________________________________

## Next Steps

1. **Set OpenAI key** (if not already): `export OPENAI_API_KEY="sk-..."`
2. **Run enable script**: `./enable_ghost_chat.sh openai`
3. **Text Ghost**: "What would a Bitcoin drop do to WOLF?"
4. **Get AI analysis**: Ghost responds with market insights!

______________________________________________________________________

## Files Created

- **`enable_ghost_chat.sh`** - One-command setup script
- **`GHOST_CHAT_SETUP.md`** - Complete setup guide
- **`GHOST_CHAT_DEMO.md`** - Example responses
- **`GHOST_CHAT_QUICKSTART.md`** - This file

______________________________________________________________________

## Example Session

**You:** What would a Bitcoin drop do to WOLF?

**Ghost:** 🤔 Thinking...

**Ghost:** 🤖 Ghost:

A Bitcoin drop would have minimal direct impact on Wolfspeed (WOLF) since they're in
different sectors—semiconductor manufacturing vs. cryptocurrency. However, there are
indirect correlations to monitor:

1. **Market Sentiment**: Crypto crashes often trigger risk-off behavior, pressuring
   growth stocks like WOLF
2. **Your Position**: Currently holding [X] shares @ $[Y]. If BTC triggers broad
   selloff, WOLF could see 3-5% pullback
3. **Current Signal**: [BUY/HOLD/SELL] with [XX]% confidence
4. **Recommendation**: Monitor NASDAQ more than Bitcoin. Set stop-loss if BTC drops
   below $50k as macro risk indicator

Bottom line: Bitcoin is a macro risk signal, not a WOLF-specific driver. Focus on chip
demand fundamentals.

______________________________________________________________________

**Ready to enable?** Run: `./enable_ghost_chat.sh openai` 🚀

# 🤖 Ghost AI Chat Setup Guide

## What Just Got Added

Ghost now has **conversational AI**capabilities! You can ask questions naturally via:

1.**HTTP API**(`/ai/chat` endpoint)
2.**Telegram**(text Ghost questions directly)

## Features

### Natural Language Q&A

Ask Ghost anything about markets and WOLF:

- "What would a Bitcoin drop do to Wolfspeed stock?"
- "Should I buy more WOLF today?"
- "How does news sentiment affect my position?"
- "What's the risk of holding through earnings?"

### Context-Aware Responses

Ghost analyzes:

- Current WOLF price & position
- News sentiment (10 most recent headlines)
- Technical indicators & signals
- Market mood & macro pressure
- Your portfolio exposure

### Existing Commands Still Work

- `/status` - Portfolio snapshot
- `/signal` - Current trading signal
- `/pnl` - Daily profit/loss
- `/help` - Command list

______________________________________________________________________

## Setup Options

### Option 1: HTTP API (No Telegram Required)**Enable AI Agent:**

```bash
export AGENTS_ENABLED=1
export AI_PROVIDER=openai  # or "ollama" for local
export OPENAI_API_KEY="your-key-here"
export AGENT_MODEL="gpt-4o-mini"  # or any model

# Restart server

pkill -f "uvicorn.*wolf_app"
nohup python -m uvicorn wolf_app:APP --host 0.0.0.0 --port 5000 > ghost_server.log 2>&1 &

```text

**Test It:**```bash

curl -X POST <<<<<http://localhost:5000/ai/chat>>>>> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -d '{"question": "What would a Bitcoin drop do to WOLF?"}'

```text

### Option 2: Telegram Chat (Recommended)**Prerequisites:**1. Telegram bot token (from [@BotFather](<<<<<https://t.me/BotFathe>>>>>r))

1. Your Telegram chat ID
2. AI agent enabled (see Option 1)**Configure Telegram:**```bash


export TELEGRAM_BOT_TOKEN="your-bot-token"
export TELEGRAM_CHAT_ID="your-chat-id"

```text**Set Webhook:**```bash

# Replace with your public URL (ngrok, Railway, etc.)

curl -X POST "<<<<<https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=https://your-domain.com/telegram/webhook">>>>>

```text**Test It:** Just open Telegram and text your bot:

```text

What would a Bitcoin drop do to WOLF?

```text

Ghost will respond with AI-powered analysis!

______________________________________________________________________

## Environment Variables Reference

### Required for AI Chat

```bash

AGENTS_ENABLED=1                    # Enable AI agent
AI_PROVIDER=openai                  # "openai" or "ollama"
OPENAI_API_KEY=sk-...              # OpenAI API key
AGENT_MODEL=gpt-4o-mini            # Model name

```text

### Optional for Telegram

```bash

TELEGRAM_BOT_TOKEN=123:ABC...      # Bot token from @BotFather
TELEGRAM_CHAT_ID=123456789         # Your Telegram user/chat ID

```text

### Already Set

```bash

GHOST_API_TOKEN=supersecret123jamaica713  # API auth token

```text

______________________________________________________________________

## Quick Start Guide

### 1. Enable AI (Using OpenAI)

Add to your `.env` or export:

```bash

export AGENTS_ENABLED=1
export AI_PROVIDER=openai
export OPENAI_API_KEY="$(railway variables get OPENAI_API_KEY)"
export AGENT_MODEL="gpt-4o-mini"

```text

### 2. Restart Ghost

```bash

pkill -f "uvicorn.*wolf_app"
cd /workspaces/GHOST
source .venv/bin/activate
nohup python -m uvicorn wolf_app:APP --host 0.0.0.0 --port 5000 > ghost_server.log 2>&1 &

```text

### 3. Test via HTTP

```bash

curl -X POST <<<<<http://localhost:5000/ai/chat>>>>> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer supersecret123jamaica713" \
  -d '{
    "question": "What would a Bitcoin drop do to Wolfspeed stock?",
    "include_context": false
  }' | jq -r '.answer'

```text

### 4. (Optional) Set Up Telegram

**Get Bot Token:**1. Message [@BotFather](<<<<<https://t.me/BotFathe>>>>>r) on Telegram

1. Send `/newbot` and follow prompts
2. Copy the token (looks like `123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11`)**Get Your Chat ID:**```bash


# Message your bot first, then

curl "<<<<<https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getUpdates">>>>> | jq '.result[0].message.chat.id'

```text**Configure:**```bash

export TELEGRAM_BOT_TOKEN="$(railway variables get TELEGRAM_BOT_TOKEN)"
export TELEGRAM_CHAT_ID="$(railway variables get TELEGRAM_CHAT_ID)"

```text**Set Webhook (if public):**```bash

curl -X POST "<<<<<https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=https://your-domain.com/telegram/webhook">>>>>

```text

______________________________________________________________________

## Example Questions

### Market Analysis

- "What would a Bitcoin drop do to WOLF?"
- "How does semiconductor demand affect WOLF stock?"
- "Is WOLF oversold right now?"


### Position Management

- "Should I buy more WOLF today?"
- "When should I take profits?"
- "What's my risk exposure?"


### News Impact

- "How is the latest news affecting WOLF?"
- "What does negative sentiment mean for my position?"
- "Should I hold through earnings?"


### Technical Analysis

- "Is WOLF showing bullish signals?"
- "What's the current trend?"
- "Is this a good entry point?"


______________________________________________________________________

## Architecture

```text

┌─────────────────┐
│  Telegram App   │
│  (You ask Q)    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────┐
│  Ghost Server (wolf_app.py)         │
│                                      │
│  1. Receive question via webhook    │
│  2. Build market context:           │
│     - Current WOLF price/position   │
│     - News sentiment (last 10)      │
│     - Technical signals             │
│     - Macro pressure                │
│  3. Send to AI with context         │
│  4. Return AI answer via Telegram   │
└────────┬────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  OpenAI API     │
│  (gpt-4o-mini)  │
└─────────────────┘

```text

______________________________________________________________________

## Troubleshooting

### "AI agent not enabled"

- Set `AGENTS_ENABLED=1`
- Configure `AI_PROVIDER` and `OPENAI_API_KEY`
- Restart server


### "missing bearer token"

- Add header: `-H "Authorization: Bearer $GHOST_API_TOKEN"`
- Token is: `supersecret123jamaica713`


### "Telegram webhook error"

- Verify `TELEGRAM_BOT_TOKEN` is set
- Check webhook is pointing to your public URL
- Test locally first with HTTP API


### "OpenAI API error"

- Verify your API key is valid and has credits
- Check model name is correct (`gpt-4o-mini`, `gpt-4`, etc.)
- Review `ghost_server.log` for details


______________________________________________________________________

## Cost Estimate

Using OpenAI GPT-4o-mini:

- ~$0.15 per 1M input tokens
- ~$0.60 per 1M output tokens
- Each question: ~500 tokens in, ~200 tokens out


-**Cost per question: ~$0.0001 (0.01¢)**Very affordable for conversational trading advice! 🚀

______________________________________________________________________

## Security Notes

1.**Webhook Validation**: Consider adding `X-Telegram-Bot-Api-Secret-Token` validation

   (see `GHOST_ARSENAL_AUDIT.md` item GH-AUD-007)

1. **Rate Limiting**: Ghost already has rate limiting on API endpoints
2. **Auth Token**: Keep `GHOST_API_TOKEN` secret
3. **API Keys**: Never commit `OPENAI_API_KEY` to git


______________________________________________________________________

## Next Steps

1. Enable AI agent (see Quick Start)
2. Test via HTTP API
3. (Optional) Set up Telegram webhook
4. Start asking Ghost questions!


**Example:**

```bash

# After setup

curl -X POST <<<<<http://localhost:5000/ai/chat>>>>> \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer supersecret123jamaica713" \
  -d '{"question": "Should I hold WOLF through earnings?"}' | jq -r '.answer'

```text

______________________________________________________________________

## What Ghost Analyzes

When you ask a question, Ghost considers:

### Price Data

- Current WOLF price
- Previous close
- Your average cost
- Current P&L


### Position Data

- Number of shares held
- Portfolio value
- Risk exposure


### News Sentiment

- Last 10 headlines
- Sentiment scores
- News signal strength


### Technical Indicators

- Current signal (BUY/SELL/HOLD)
- Signal confidence
- Mode (price-driven, news-driven, hybrid)


### Macro Context

- Market mood (if Stage1 enabled)
- Macro pressure metrics
- World events context


### Module Weights

- Which signals Ghost trusts most
- Dynamic weight adjustments


Ghost synthesizes all this into actionable advice! 🎯

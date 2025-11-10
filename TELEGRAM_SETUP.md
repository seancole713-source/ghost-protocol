# Telegram Integration Setup

## Current Status

- Telegram feed: Currently disabled
- Bot configured in env vars
- Needs activation in Railway

## Required Environment Variables (Already in railway_env_vars.txt)

```bash
TELEGRAM_BOT_TOKEN=8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw
TELEGRAM_CHAT_ID=940596997
```

## Activation Steps

### 1. Verify Variables in Railway

- Go to Railway dashboard → Project → Variables
- Confirm both `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` are set
- They should already be there from railway_env_vars.txt

### 2. Test Telegram Endpoint

```bash
curl -X POST https://web-production-8e9a0.up.railway.app/api/telegram/test
```

Expected response:

```json
{
  "ok": true,
  "sent": true,
  "can_send": true,
  "card": "... portfolio card ..."
}
```

### 3. Test Telegram Commands

Send these messages to your bot in Telegram:

- `/status` - Get portfolio status
- `/help` - List available commands
- `What is Bitcoin's price?` - AI Q&A
- `Should I buy WOLF?` - Get trading advice

### 4. Verify in Cockpit

```bash
curl https://web-production-8e9a0.up.railway.app/api/cockpit | jq '.status.feeds.telegram'
```

Should return: `true`

## Troubleshooting

If Telegram feed shows `false`:

1. Check Railway logs for: "Telegram bot initialized"
2. Verify token is valid: `curl https://api.telegram.org/bot<TOKEN>/getMe`
3. Ensure chat_id is correct (your Telegram user ID)
4. Redeploy if needed

## Features Enabled

Once active, Telegram provides:

- ✅ Real-time portfolio updates
- ✅ Price alerts
- ✅ AI-powered Q&A
- ✅ Trading recommendations
- ✅ News summaries
- ✅ Crypto & stock data on demand

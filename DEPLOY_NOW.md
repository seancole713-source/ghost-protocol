# 🚀 Quick Fix: Deploy AI Chat to Railway

## Why You're Still Getting Old Response

Your Telegram webhook points to **Railway** (old code), not your local server (new
code).

## One-Command Solution

```bash
./deploy_complete.sh
```

This will:

1. Push code to Railway
2. Update Telegram webhook
3. Test deployment
4. Show you next steps for env vars

## After Deploy

Set these in Railway dashboard:

```
AGENTS_ENABLED=1
AI_PROVIDER=openai
AGENT_MODEL=gpt-4o-mini
OPENAI_API_KEY=your-key-here
```

Then text your bot: **"What would a Bitcoin drop do to WOLF?"**

You'll get AI analysis! 🎉

______________________________________________________________________

## What's Fixed

✅ Telegram AI chat (natural questions)\
✅ Test button (GET/POST, no auth)\
✅ Direct alert sending

## Ready?

```bash
./deploy_complete.sh
```

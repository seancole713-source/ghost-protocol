# 🔍 Railway Environment Variables Check

## Required for AI Chat to Work

Based on your Railway secrets, verify these are set to the correct **VALUES** (not just
`*******`):

### ✅ Core Requirements

| Variable | Required Value | Purpose | |----------|---------------|---------| |
`AGENTS_ENABLED` | `1` | Enable AI agent system | | `AI_PROVIDER` | `openai` | Use
OpenAI (not ollama) | | `AGENT_MODEL` | `gpt-4o-mini` | Model to use | |
`OPENAI_API_KEY` | `sk-proj-...` | Your OpenAI API key | | `TELEGRAM_BOT_TOKEN` |
`123:ABC...` | Your bot token | | `TELEGRAM_CHAT_ID` | `123456789` | Your chat ID |

### 📋 How to Verify

Go to Railway dashboard and check each variable has the **correct value**, not just
`*******`.

**Click on each variable to see its actual value.**

### ⚠️ Common Issues

1. **`AI_PROVIDER` set to `ollama`** ❌

   - Change to: `openai`

2. **`AGENT_MODEL` not set or wrong** ❌

   - Set to: `gpt-4o-mini` (cheaper) or `gpt-4` (better)

3. **`AGENTS_ENABLED` set to `0`** ❌

   - Change to: `1`

### 🎯 Quick Fix Checklist

In Railway dashboard, make sure:

- [ ] `AGENTS_ENABLED` = `1`
- [ ] `AI_PROVIDER` = `openai`
- [ ] `AGENT_MODEL` = `gpt-4o-mini`
- [ ] `OPENAI_API_KEY` = `sk-proj-...` (your actual key)
- [ ] `TELEGRAM_BOT_TOKEN` = your bot token
- [ ] `TELEGRAM_CHAT_ID` = your chat ID

### 🚀 After Fixing

1. Railway will auto-redeploy (takes ~2 min)
2. Text your bot: "What would Bitcoin drop do to WOLF?"
3. Get AI response! 🎉

______________________________________________________________________

## Test Railway Deployment

Once Railway redeploys, test it:

```bash
# Test health
curl https://web-production-8e9a0.up.railway.app/health

# Test alerts
curl https://web-production-8e9a0.up.railway.app/alerts/selftest

# Test AI chat (needs auth)
curl -X POST https://web-production-8e9a0.up.railway.app/ai/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $GHOST_API_TOKEN" \
  -d '{"question": "Hello"}' | jq -r '.answer'
```

If AI chat returns actual text (not "AI agent not enabled"), it's working!

______________________________________________________________________

## Still Not Working?

Check Railway logs:

1. Go to Railway dashboard
2. Click on your deployment
3. View logs
4. Look for errors about AGENTS_ENABLED or AI_PROVIDER

Common log errors:

- `"AI agent not enabled"` → Set `AGENTS_ENABLED=1`
- `"No module named 'ollama'"` → Set `AI_PROVIDER=openai`
- `"OpenAI API error"` → Check `OPENAI_API_KEY` is valid

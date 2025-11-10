# 🚂 RAILWAY ENV VARS - COPY/PASTE READY

## ✅ ADD THESE (Missing Required Variables)

```bash
# AgentKit (NEW - fully implemented)
AGENTKIT_ENABLED=true
AGENTS_ENABLED=true
AGENT_MODEL=gpt-4o-mini

# Research Enrichment
RESEARCH_LLM_ON=1
RESEARCH_LLM_MODEL=gpt-4o-mini

# Portfolio (UPDATE WITH YOUR VALUES)
WOLF_QTY=8.41959051
WOLF_AVG_COST=359.28
WOLF_PERSIST_MODE=sqlite

# Security
CSP_MODE=prod
ALLOWED_ORIGINS=https://your-ghost-railway-domain.railway.app
```

______________________________________________________________________

## ❌ DELETE THESE (Placeholders - Do Nothing)

```bash
# Agent placeholders (never implemented)
AGENT_ROLE
AGENT_POLICY
AGENT_ENDPOINT_URL

# OpenAI extras (not used)
OPENAI_ORG_ID

# Vector DB (not implemented)
VECTOR_DB_URL
VECTOR_DB_API_KEY
VECTOR_STORE_ID
VECTOR_SOURCE
MEMORY_TTL_DAYS

# Generic cache (not used)
CACHE_MODE
CACHE_TTL
SYSTEM_MODE

# Model failover (not implemented)
MODEL_FALLBACK_ENABLED
MODEL_FALLBACK_CHAIN
MODEL_FAIL_RETRY
MODEL_FAIL_DELAY_SEC

# Auto-fixer (not implemented)
AUTO_FIXER_ENABLED
AUTO_FIX_INTERVAL_SEC
AUTO_RESTART_COOLDOWN_SEC

# Other unused
DATA_FRESHNESS_SEC
ALERT_CHANNEL
```

______________________________________________________________________

## ✅ KEEP THESE (Already Have - Working)

```bash
# Core
SIM_MODE=0
GHOST_API_TOKEN=your_token
GHOST_FOCUS_TICKER=WOLF

# API Keys
POLYGON_API_KEY=8VIvELVXiLG30K2l1348RzSurffLM0jR
ALPHAVANTAGE_API_KEY=3WNNLA81KS7BG4AK

# OpenAI
OPENAI_API_KEY=sk-proj-...
OPENAI_AGENT_API_KEY=sk-proj-...

# Telegram
TELEGRAM_BOT_TOKEN=8229069551:AAE...
TELEGRAM_CHAT_ID=940596997
```

______________________________________________________________________

## 🔧 OPTIONAL (Nice To Have)

```bash
# News
NEWS_SENTIMENT_ON=1
REUTERS_FEEDS_ON=1

# Logging
LOG_LEVEL=INFO
LOG_JSON=1

# Monitoring
PROMETHEUS_MULTIPROC_DIR=/tmp/ghost_prom
```

______________________________________________________________________

## 📋 QUICK RAILWAY COMMANDS

### Add Missing Variables:

```bash
railway variables set AGENTKIT_ENABLED=true
railway variables set AGENTS_ENABLED=true
railway variables set AGENT_MODEL=gpt-4o-mini
railway variables set RESEARCH_LLM_ON=1
railway variables set RESEARCH_LLM_MODEL=gpt-4o-mini
railway variables set WOLF_QTY=8.41959051
railway variables set WOLF_AVG_COST=359.28
railway variables set WOLF_PERSIST_MODE=sqlite
railway variables set CSP_MODE=prod
railway variables set ALLOWED_ORIGINS=https://your-domain.railway.app
```

### Delete Placeholders:

```bash
railway variables delete AGENT_ROLE
railway variables delete AGENT_POLICY
railway variables delete AGENT_ENDPOINT_URL
railway variables delete OPENAI_ORG_ID
railway variables delete VECTOR_DB_URL
railway variables delete VECTOR_DB_API_KEY
railway variables delete VECTOR_STORE_ID
railway variables delete VECTOR_SOURCE
railway variables delete MEMORY_TTL_DAYS
railway variables delete CACHE_MODE
railway variables delete CACHE_TTL
railway variables delete SYSTEM_MODE
railway variables delete MODEL_FALLBACK_ENABLED
railway variables delete MODEL_FALLBACK_CHAIN
railway variables delete MODEL_FAIL_RETRY
railway variables delete MODEL_FAIL_DELAY_SEC
railway variables delete AUTO_FIXER_ENABLED
railway variables delete AUTO_FIX_INTERVAL_SEC
railway variables delete DATA_FRESHNESS_SEC
railway variables delete AUTO_RESTART_COOLDOWN_SEC
railway variables delete ALERT_CHANNEL
```

### Verify:

```bash
railway variables
```

______________________________________________________________________

## 💡 TIP: AgentKit ON vs OFF

**Want persistent AI memory?**

```bash
AGENTKIT_ENABLED=true   # Uses OpenAI Assistants API (stateful)
```

**Want simple/fast decisions?**

```bash
AGENTKIT_ENABLED=false  # Uses chat completions (stateless)
```

Both work - pick what fits your needs!

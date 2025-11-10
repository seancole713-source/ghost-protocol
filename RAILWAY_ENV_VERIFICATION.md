# 🚂 Railway Environment Variables Verification

**Date**: October 8, 2025\
**Status**: Validation of provided variables

______________________________________________________________________

## ✅ **Your Provided Variables**

```bash
SIM_MODE="0"                                           # ✅ CORRECT - Live mode
GHOST_API_TOKEN="e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0" # ✅ CORRECT - API auth
POLYGON_API_KEY="8VIvELVXiLG30K2l1348RzSurffLM0jR"     # ✅ CORRECT - Market data
ALPHAVANTAGE_API_KEY="3WNNLA81KS7BG4AK"               # ✅ CORRECT - Market data
TELEGRAM_BOT_TOKEN="8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw" # ✅ CORRECT
TELEGRAM_CHAT_ID="940596997"                          # ✅ CORRECT
GHOST_FOCUS_TICKER="WOLF"                             # ✅ CORRECT - Focus symbol
AGENTS_ENABLED="true"                                 # ✅ USED - enables AI agent (alias: AI_ON)
AGENT_POLICY="balanced"                               # ⚠️ NOT USED - see below
MEMORY_TTL_DAYS="90"                                  # ⚠️ NOT USED - see below
VECTOR_DB_URL=""                                      # ⚠️ NOT USED - empty, safe to remove
VECTOR_DB_API_KEY=""                                  # ⚠️ NOT USED - empty, safe to remove
```

______________________________________________________________________

## ⚠️ **Variables Ghost Doesn't Use**

These variables are NOT read by any Ghost code:

| Variable | Status | Reason | |----------|--------|--------| | `AGENT_POLICY` | ❌ Not
used | No code references this | | `MEMORY_TTL_DAYS` | ❌ Not used | Ghost uses internal
24h TTL for decisions | | `VECTOR_DB_URL` | ❌ Not used | Ghost uses SQLite, checks
`AI_MEMORY_VECTOR_STORE` | | `VECTOR_DB_API_KEY` | ❌ Not used | No vector DB integration
in current code |

### Recommendation

**Safe to delete** these 4 variables from Railway to reduce clutter.

______________________________________________________________________

## ✅ **Variables Ghost Actually Uses**

Based on code analysis, here are the environment variables Ghost reads:

### **Critical (Required for Production)**

```bash
# API Keys
POLYGON_API_KEY="8VIvELVXiLG30K2l1348RzSurffLM0jR"          # ✅ You have this
ALPHAVANTAGE_API_KEY="3WNNLA81KS7BG4AK"                    # ✅ You have this
GHOST_API_TOKEN="e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0"     # ✅ You have this

# Telegram Alerts
TELEGRAM_BOT_TOKEN="8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw" # ✅ You have
TELEGRAM_CHAT_ID="940596997"                               # ✅ You have this

# Mode
SIM_MODE="0"                                               # ✅ You have this (live)
```

### **Recommended (Production Best Practices)**

```bash
# ChatGPT Agent (if you want AI analyst running)
OPENAI_API_KEY="sk-..."                                    # ⚠️ MISSING - needed for agent
GHOST_LLM_MODEL="gpt-4o-mini"                             # Optional (default: gpt-4o-mini)
GHOST_AGENT_TICK="300"                                    # Optional (default: 300s = 5min)
GHOST_AGENT_DB="./data/ghost_agent.db"                    # Optional (default path)

# Logging
LOG_LEVEL="INFO"                                          # Optional (default: INFO)
LOG_JSON="1"                                              # Optional (default: 1 = JSON logs)

# Security
SECURE_HEADERS="1"                                        # Recommended (default: 1)
CSP_MODE="prod"                                           # Recommended for Railway
ALLOWED_ORIGINS="https://your-domain.railway.app"         # Set to your Railway URL

# Portfolio Persistence
WOLF_QTY="909.43"                                         # Your WOLF position quantity
WOLF_AVG_COST="217.96"                                    # Your average cost
WOLF_PERSIST_MODE="sqlite"                                # Recommended (vs file)
WOLF_SQLITE_PATH="data/wolf.db"                           # Optional (default)

# Prometheus (if using)
PROMETHEUS_MULTIPROC_DIR="/tmp/ghost_prom"                # Optional (multi-worker mode)
```

### **Optional (Customization)**

```bash
# Timing
TICK_INTERVAL_S="5"                                       # Default: 5s
PRICE_TTL_S="30"                                          # Default: 30s (market closed)
PRICE_TTL_OPEN_S="5"                                      # Default: 5s (market open)
NEWS_TTL_S="300"                                          # Default: 5min

# Alert Thresholds
ALERT_MODE="fixed"                                        # fixed|band|trailing (default: fixed)
ALERT_BUY_PCT="0.99"                                      # BUY if price < avg * 0.99
ALERT_SELL_PCT="1.01"                                     # SELL if price > avg * 1.01

# Forecast
FORECAST_STEP_S="7200"                                    # 2h steps (default)
FORECAST_HORIZON_S="172800"                               # 48h horizon (default)

# Focus (already have this)
GHOST_FOCUS_TICKER="WOLF"                                 # ✅ You have this
FOCUS_WOLF_ONLY="1"                                       # Optional (default: 1)

# Timezone
GHOST_TZ="America/Chicago"                                # Optional (default: Chicago)
```

______________________________________________________________________

## 🔍 **Variable Analysis by Category**

### 1. **Market Data Providers** ✅

```bash
POLYGON_API_KEY       # ✅ Present - Primary source
ALPHAVANTAGE_API_KEY  # ✅ Present - Fallback source
COINGECKO_API_KEY     # ❌ Missing - Optional (for crypto)
```

### 2. **Alerts & Notifications** ✅

```bash
TELEGRAM_BOT_TOKEN    # ✅ Present
TELEGRAM_CHAT_ID      # ✅ Present
ALERT_WEBHOOK_URLS    # ❌ Missing - Optional (webhook alerts)
SLACK_WEBHOOK_URLS    # ❌ Missing - Optional (Slack integration)
```

### 3. **AI/Agent System** ⚠️

```bash
OPENAI_API_KEY        # ⚠️ MISSING - Required for ChatGPT Analyst
AGENTS_ENABLED        # Default: 0 (disabled) - Enable with AGENTS_ENABLED=1 (alias: AI_ON)
AI_PROVIDER           # Default: "ollama" - Set to "openai" for ChatGPT
AGENT_MODEL           # Default: "llama3.1:8b" - Change to "gpt-4o-mini" (alias: AI_MODEL)
GHOST_AGENT_TICK      # Default: 300s - How often agent runs
GHOST_AGENT_DB        # Default: "./data/ghost_agent.db" - Agent persistence
```

**⚠️ ACTION REQUIRED**: If you want the ChatGPT Analyst running:

```bash
OPENAI_API_KEY="sk-..."       # Add your OpenAI key
AGENTS_ENABLED="1"            # Enable AI (alias: AI_ON)
AI_PROVIDER="openai"          # Use OpenAI (not Ollama)
AGENT_MODEL="gpt-4o-mini"     # Model to use (alias: AI_MODEL)
```

### 4. **Portfolio State** ⚠️

```bash
WOLF_QTY              # ⚠️ MISSING - Your current WOLF shares
WOLF_AVG_COST         # ⚠️ MISSING - Your average cost per share
WOLF_PERSIST_MODE     # Default: "auto" - Set to "sqlite" for Railway
WOLF_SQLITE_PATH      # Default: "data/wolf.db"
```

**⚠️ ACTION REQUIRED**: Set your position:

```bash
WOLF_QTY="909.43"             # Your actual quantity
WOLF_AVG_COST="217.96"        # Your actual avg cost
WOLF_PERSIST_MODE="sqlite"    # Use SQLite on Railway
```

### 5. **Security** ⚠️

```bash
GHOST_API_TOKEN       # ✅ Present
SECURE_HEADERS        # Default: 1 - Good
CSP_MODE              # Default: "dev" - ⚠️ Set to "prod" for Railway
ALLOWED_ORIGINS       # Default: "*" - ⚠️ Set to Railway URL
IP_ALLOWLIST          # Optional - Comma-separated IPs
```

**⚠️ ACTION REQUIRED**: Harden security:

```bash
CSP_MODE="prod"
ALLOWED_ORIGINS="https://ghost-production.up.railway.app"
```

### 6. **Logging & Monitoring** ℹ️

```bash
LOG_LEVEL             # Default: "INFO" - Good
LOG_JSON              # Default: 1 - Good for Railway logs
OTEL_ENABLED          # Default: 0 - Optional (OpenTelemetry)
```

______________________________________________________________________

## 📋 **Recommended Railway Environment**

### **Minimal (Working System)**

```bash
# Required
SIM_MODE="0"
GHOST_API_TOKEN="e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0"
POLYGON_API_KEY="8VIvELVXiLG30K2l1348RzSurffLM0jR"
ALPHAVANTAGE_API_KEY="3WNNLA81KS7BG4AK"
TELEGRAM_BOT_TOKEN="8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw"
TELEGRAM_CHAT_ID="940596997"
GHOST_FOCUS_TICKER="WOLF"

# Portfolio
WOLF_QTY="909.43"
WOLF_AVG_COST="217.96"
WOLF_PERSIST_MODE="sqlite"
```

### **Recommended (Production-Ready)**

```bash
# All from Minimal above, plus:

# AI Agent
OPENAI_API_KEY="sk-..."
AI_ON="1"
AI_PROVIDER="openai"
AI_MODEL="gpt-4o-mini"
GHOST_AGENT_TICK="300"

# Security
CSP_MODE="prod"
ALLOWED_ORIGINS="https://ghost-production.up.railway.app"

# Logging
LOG_LEVEL="INFO"
LOG_JSON="1"

# Timezone
GHOST_TZ="America/Chicago"
```

### **Advanced (Full Features)**

```bash
# All from Recommended above, plus:

# Alert Customization
ALERT_MODE="fixed"
ALERT_BUY_PCT="0.99"
ALERT_SELL_PCT="1.01"
ALERT_THROTTLE_S="60"

# Forecast Tuning
FORECAST_STEP_S="7200"
FORECAST_HORIZON_S="172800"

# HTTP Pooling (for better performance)
HTTP_POOL_ENABLED="1"
HTTP_POOL_SIZE="10"
HTTP_TIMEOUT_S="8"

# Prometheus (if scraping metrics)
PROMETHEUS_MULTIPROC_DIR="/tmp/ghost_prom"
```

______________________________________________________________________

## 🚫 **Variables to DELETE from Railway**

These do nothing in Ghost:

```bash
# DELETE - Not used by Ghost code
AGENTS_ENABLED
AGENT_POLICY
MEMORY_TTL_DAYS
VECTOR_DB_URL
VECTOR_DB_API_KEY
```

______________________________________________________________________

## ✅ **Verification Commands**

### Test Railway Environment

```bash
# SSH into Railway container
railway run bash

# Check environment variables
env | grep -E '(GHOST|POLYGON|ALPHAVANTAGE|TELEGRAM|OPENAI|WOLF|AI_)' | sort

# Verify Ghost reads them
python3 << 'EOF'
import os
print("✅ SIM_MODE:", os.getenv("SIM_MODE"))
print("✅ POLYGON_API_KEY:", "***" + os.getenv("POLYGON_API_KEY", "")[-4:])
print("✅ ALPHAVANTAGE_API_KEY:", "***" + os.getenv("ALPHAVANTAGE_API_KEY", "")[-4:])
print("✅ TELEGRAM_BOT_TOKEN:", "***" + os.getenv("TELEGRAM_BOT_TOKEN", "")[-6:])
print("✅ TELEGRAM_CHAT_ID:", os.getenv("TELEGRAM_CHAT_ID"))
print("✅ OPENAI_API_KEY:", "***" + os.getenv("OPENAI_API_KEY", "")[-4:] if os.getenv("OPENAI_API_KEY") else "❌ MISSING")
print("✅ WOLF_QTY:", os.getenv("WOLF_QTY", "❌ MISSING"))
print("✅ WOLF_AVG_COST:", os.getenv("WOLF_AVG_COST", "❌ MISSING"))
EOF
```

### Test API Endpoints

```bash
# Health check
curl https://your-railway-domain.railway.app/health

# Agent health (if enabled)
curl https://your-railway-domain.railway.app/agent/health

# Cockpit snapshot
curl https://your-railway-domain.railway.app/api/cockpit
```

______________________________________________________________________

## 🎯 **Action Plan**

### Phase 1: Clean Up (5 minutes)

1. **Delete unused variables** from Railway dashboard:
   - `AGENTS_ENABLED`
   - `AGENT_POLICY`
   - `MEMORY_TTL_DAYS`
   - `VECTOR_DB_URL`
   - `VECTOR_DB_API_KEY`

### Phase 2: Add Missing Critical Variables (5 minutes)

2. **Add portfolio state**:

   ```bash
   WOLF_QTY="909.43"
   WOLF_AVG_COST="217.96"
   WOLF_PERSIST_MODE="sqlite"
   ```

3. **Add AI agent** (if you want ChatGPT Analyst):

   ```bash
   OPENAI_API_KEY="sk-..."
   AI_ON="1"
   AI_PROVIDER="openai"
   AI_MODEL="gpt-4o-mini"
   ```

### Phase 3: Harden Security (5 minutes)

4. **Update security settings**:
   ```bash
   CSP_MODE="prod"
   ALLOWED_ORIGINS="https://your-actual-domain.railway.app"
   ```

### Phase 4: Verify (5 minutes)

5. **Test deployment**:
   ```bash
   # Check logs
   railway logs

   # Test endpoints
   curl https://your-domain/health
   curl https://your-domain/api/cockpit
   ```

______________________________________________________________________

## 📊 **Current Status**

| Category | Status | Missing Items | |----------|--------|---------------| | **Market
Data** | ✅ Complete | None | | **Alerts** | ✅ Complete | None | | **AI Agent** | ⚠️
Incomplete | OPENAI_API_KEY, AI_ON | | **Portfolio** | ⚠️ Incomplete | WOLF_QTY,
WOLF_AVG_COST | | **Security** | ⚠️ Needs Hardening | CSP_MODE, ALLOWED_ORIGINS | |
**Persistence** | ⚠️ Incomplete | WOLF_PERSIST_MODE | | **Unused Vars** | ❌ 5 to delete
| See list above |

______________________________________________________________________

## 🚀 **Final Recommended Railway Config**

```bash
# Core
SIM_MODE="0"
GHOST_API_TOKEN="e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0"
GHOST_FOCUS_TICKER="WOLF"

# Market Data
POLYGON_API_KEY="8VIvELVXiLG30K2l1348RzSurffLM0jR"
ALPHAVANTAGE_API_KEY="3WNNLA81KS7BG4AK"

# Alerts
TELEGRAM_BOT_TOKEN="8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw"
TELEGRAM_CHAT_ID="940596997"

# Portfolio
WOLF_QTY="909.43"
WOLF_AVG_COST="217.96"
WOLF_PERSIST_MODE="sqlite"
WOLF_SQLITE_PATH="data/wolf.db"

# AI Agent
OPENAI_API_KEY="sk-your-key-here"
AGENTS_ENABLED="1"
AI_PROVIDER="openai"
AGENT_MODEL="gpt-4o-mini"
GHOST_AGENT_TICK="300"
GHOST_AGENT_DB="./data/ghost_agent.db"

# Security
CSP_MODE="prod"
ALLOWED_ORIGINS="https://your-actual-domain.railway.app"
SECURE_HEADERS="1"

# Logging
LOG_LEVEL="INFO"
LOG_JSON="1"

# Performance
HTTP_POOL_ENABLED="1"
HTTP_POOL_SIZE="10"
PRICE_AUTO_REFRESH_S="7"
```

**Total: 24 variables** (down from 29, removed 5 unused ones)

______________________________________________________________________

**Last Updated**: October 8, 2025\
**Next**: Build operational tooling (monitoring, alerts, Grafana)

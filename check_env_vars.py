#!/usr/bin/env python3
"""Check Ghost Protocol environment variables for issues"""

# Your environment variables
env_vars = {
    "ADMIN_IP_ALLOWLIST": "127.0.0.1,::1,0.0.0.0/0",
    "AGENT_ENDPOINT_URL": "https://api.openai.com/v1/chat/completions",
    "AGENT_MODEL": "gpt-4o-mini",
    "ALPACA_PAPER": "1",
    "ALPHAVANTAGE_API_KEY": "3WNNLA81KS7BG4AK",
    "BROKER": "alpaca",
    "CRYPTO_ENABLED": "1",
    "DISABLE_PREDICTION_AUTH": "1",
    "FOCUS_WOLF_ONLY": "0",
    "LOG_LEVEL": "INFO",
    "POLYGON_API_KEY": "8VIvELVXiLG30K2l1348RzSurffLM0jR",
    "TELEGRAM_BOT_TOKEN": "8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw",
    "TELEGRAM_CHAT_ID": "940596997",
    "ALERT_STYLE": "simple",
    "ALERT_SIMPLE_FORMAT": "balanced",
    "MIN_ALERT_CONFIDENCE": "0.60",
    "PRICE_SOURCE_PRIMARY": "polygon",
    "PRICE_SOURCE_SECONDARY": "yahoo",
    "PRICE_FRESHNESS_THRESHOLD_S": "300",
    "PRICE_STALENESS_SECONDS": "300",
    "REDIS_URL": "rediss://default:AVriAAIncDJmNmUyNjFmMDRkMDE0YzE2OWNiOTY0MmYxZjcxMWYxNXAyMjMyNjY@comic-hookworm-23266.upstash.io:6379/0",
    "GHOST_API_TOKEN": "edaa4eac-6455-4693-a745-142cb6deef03",
    "OPENAI_API_KEY": "sk-proj-[REDACTED]",
}

issues = []
warnings = []
ok = []

print("🔍 CHECKING GHOST PROTOCOL ENVIRONMENT\n")
print("=" * 70)

# 1. Critical API Keys
if env_vars.get("OPENAI_API_KEY", "").startswith("sk-proj-"):
    ok.append("✅ OPENAI_API_KEY: Present (sk-proj-...)")
else:
    issues.append("❌ OPENAI_API_KEY: Missing or wrong format (need sk-proj-...)")

if env_vars.get("POLYGON_API_KEY") and len(env_vars["POLYGON_API_KEY"]) > 20:
    ok.append("✅ POLYGON_API_KEY: Present and valid length")
else:
    issues.append("❌ POLYGON_API_KEY: Missing or too short")

if env_vars.get("ALPHAVANTAGE_API_KEY") and len(env_vars["ALPHAVANTAGE_API_KEY"]) > 10:
    ok.append("✅ ALPHAVANTAGE_API_KEY: Present (fallback provider)")
else:
    warnings.append("⚠️  ALPHAVANTAGE_API_KEY: Missing (optional, but good for redundancy)")

# 2. Telegram Configuration
if env_vars.get("TELEGRAM_BOT_TOKEN") and ":" in env_vars["TELEGRAM_BOT_TOKEN"]:
    ok.append("✅ TELEGRAM_BOT_TOKEN: Valid format (bot_id:token)")
else:
    issues.append("❌ TELEGRAM_BOT_TOKEN: Missing or invalid format")

if env_vars.get("TELEGRAM_CHAT_ID") and env_vars["TELEGRAM_CHAT_ID"].isdigit():
    ok.append("✅ TELEGRAM_CHAT_ID: Valid numeric ID")
else:
    issues.append("❌ TELEGRAM_CHAT_ID: Missing or invalid")

# 3. Ghost Hunter Phase 1 Settings
if env_vars.get("ALERT_STYLE") in ["simple", "verbose"]:
    ok.append(f"✅ ALERT_STYLE: '{env_vars['ALERT_STYLE']}' (Cash-App style enabled)")
else:
    issues.append(f"❌ ALERT_STYLE: '{env_vars.get('ALERT_STYLE')}' (must be 'simple' or 'verbose')")

if env_vars.get("ALERT_SIMPLE_FORMAT") in ["compact", "balanced", "context"]:
    ok.append(f"✅ ALERT_SIMPLE_FORMAT: '{env_vars['ALERT_SIMPLE_FORMAT']}' (140 char format)")
else:
    warnings.append(f"⚠️  ALERT_SIMPLE_FORMAT: '{env_vars.get('ALERT_SIMPLE_FORMAT')}' (use compact/balanced/context)")

try:
    conf = float(env_vars.get("MIN_ALERT_CONFIDENCE", "0"))
    if 0 <= conf <= 1:
        ok.append(f"✅ MIN_ALERT_CONFIDENCE: {conf} ({int(conf*100)}% threshold)")
    else:
        issues.append(f"❌ MIN_ALERT_CONFIDENCE: {conf} (must be 0.0-1.0)")
except ValueError:
    issues.append("❌ MIN_ALERT_CONFIDENCE: Not a valid number")

# 4. Price Provider Settings
if env_vars.get("PRICE_SOURCE_PRIMARY") in ["polygon", "yahoo", "alphavantage"]:
    ok.append(f"✅ PRICE_SOURCE_PRIMARY: {env_vars['PRICE_SOURCE_PRIMARY']}")
else:
    issues.append(f"❌ PRICE_SOURCE_PRIMARY: Invalid provider")

if env_vars.get("PRICE_SOURCE_SECONDARY") in ["polygon", "yahoo", "alphavantage"]:
    ok.append(f"✅ PRICE_SOURCE_SECONDARY: {env_vars['PRICE_SOURCE_SECONDARY']} (fallback)")
else:
    issues.append(f"❌ PRICE_SOURCE_SECONDARY: Invalid provider")

if env_vars.get("PRICE_SOURCE_PRIMARY") == env_vars.get("PRICE_SOURCE_SECONDARY"):
    warnings.append("⚠️  PRIMARY and SECONDARY price sources are the same (no real fallback)")

try:
    fresh = int(env_vars.get("PRICE_FRESHNESS_THRESHOLD_S", "0"))
    if fresh > 0:
        ok.append(f"✅ PRICE_FRESHNESS_THRESHOLD_S: {fresh}s ({fresh//60}min staleness check)")
    else:
        warnings.append("⚠️  PRICE_FRESHNESS_THRESHOLD_S: Not set or 0 (no staleness checks)")
except ValueError:
    issues.append("❌ PRICE_FRESHNESS_THRESHOLD_S: Not a valid number")

# 5. Duplicate/Conflicting Settings
if env_vars.get("PRICE_STALENESS_SECONDS"):
    warnings.append(f"⚠️  PRICE_STALENESS_SECONDS: Deprecated, use PRICE_FRESHNESS_THRESHOLD_S instead (both are {env_vars.get('PRICE_STALENESS_SECONDS')}s)")

# 6. Broker Settings
if env_vars.get("BROKER") == "alpaca":
    if env_vars.get("ALPACA_PAPER") == "1":
        ok.append("✅ Alpaca Broker: PAPER TRADING mode (safe)")
    else:
        warnings.append("⚠️  Alpaca Broker: LIVE TRADING mode enabled!")
else:
    warnings.append(f"⚠️  BROKER: {env_vars.get('BROKER')} (not alpaca)")

# 7. Redis
if env_vars.get("REDIS_URL", "").startswith("rediss://"):
    ok.append("✅ REDIS_URL: Configured with SSL (Upstash)")
elif env_vars.get("REDIS_URL", "").startswith("redis://"):
    warnings.append("⚠️  REDIS_URL: Not using SSL (rediss://)")
else:
    warnings.append("⚠️  REDIS_URL: Missing or invalid")

# 8. Security
if "0.0.0.0/0" in env_vars.get("ADMIN_IP_ALLOWLIST", ""):
    warnings.append("⚠️  ADMIN_IP_ALLOWLIST: Allows ALL IPs (0.0.0.0/0) - wide open!")
else:
    ok.append("✅ ADMIN_IP_ALLOWLIST: Restricted to specific IPs")

if env_vars.get("GHOST_API_TOKEN") and len(env_vars["GHOST_API_TOKEN"]) > 30:
    ok.append("✅ GHOST_API_TOKEN: Present (API protected)")
else:
    warnings.append("⚠️  GHOST_API_TOKEN: Missing or too short (API unprotected)")

if env_vars.get("DISABLE_PREDICTION_AUTH") == "1":
    warnings.append("⚠️  DISABLE_PREDICTION_AUTH=1: Predictions are PUBLIC (no auth required)")
else:
    ok.append("✅ Prediction auth: Required")

# 9. Feature Flags
if env_vars.get("CRYPTO_ENABLED") == "1":
    ok.append("✅ CRYPTO_ENABLED: Crypto predictions enabled")

if env_vars.get("FOCUS_WOLF_ONLY") == "0":
    ok.append("✅ FOCUS_WOLF_ONLY: Multi-symbol mode (not just WOLF)")
else:
    warnings.append("⚠️  FOCUS_WOLF_ONLY=1: Only WOLF predictions (restricted mode)")

# Print Results
print("\n✅ VALID CONFIGURATION")
print("=" * 70)
for item in ok:
    print(item)

if warnings:
    print("\n⚠️  WARNINGS (Review These)")
    print("=" * 70)
    for item in warnings:
        print(item)

if issues:
    print("\n❌ CRITICAL ISSUES (Fix These!)")
    print("=" * 70)
    for item in issues:
        print(item)
    print("\n🚨 Found critical issues that may prevent Ghost from working properly!")
else:
    print("\n🎉 NO CRITICAL ISSUES FOUND!")
    print("=" * 70)

print(f"\nSummary: {len(ok)} OK • {len(warnings)} Warnings • {len(issues)} Issues\n")

# Specific recommendations
if not issues:
    print("💡 RECOMMENDATIONS:")
    print("=" * 70)
    print("1. ✅ All critical settings are valid")
    print("2. ✅ Telegram alerts will work")
    print("3. ✅ Ghost Hunter Phase 1 features are configured")
    print("4. ✅ Price reliability with fallback is enabled")
    
    if warnings:
        print(f"\n   Review {len(warnings)} warnings above for optimization opportunities")

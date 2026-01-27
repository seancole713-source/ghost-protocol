#!/usr/bin/env python3
import requests
import time

messages = [
"""⚠️ *RAILWAY DEPLOYED BUT ISSUE FOUND*

Railway rebuilt (uptime: 2 min) but predictions still showing old behavior.

*Current Response:*
```json
{
  "confidence": 0.56,
  "direction": "DOWN"
}
```

❌ Missing: current_price, feature_count
❌ Confidence still low (~0.5)

━━━━━━━━━━━━━━━━━━━━
""",
"""🔍 *ROOT CAUSE*

FREE-TIER providers are in the codebase but may not be wired to the prediction API yet.

*Need to check:*
1. Is unified_provider imported in wolf_app.py?
2. Is feature_orchestrator using unified_provider?
3. Are providers initialized at startup?

━━━━━━━━━━━━━━━━━━━━
""",
"""⏭️ *NEXT STEP*

I'll verify the provider wiring and fix any import issues.

The DB persistence code IS deployed (that's good).
The evaluator IS running (that's good).

Just need to fix provider integration.

Give me 5 minutes to diagnose and fix...
"""
]

TELEGRAM_BOT_TOKEN = "8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw"
TELEGRAM_CHAT_ID = "940596997"

def send(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)

for i, msg in enumerate(messages, 1):
    print(f"Sending {i}/{len(messages)}...")
    send(msg)
    if i < len(messages):
        time.sleep(1)

print("✅ Sent")

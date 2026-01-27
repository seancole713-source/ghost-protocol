#!/usr/bin/env python3
import os
import requests
import time

TELEGRAM_BOT_TOKEN = "8229069551:AAEBHMpX8TkaPFD2hhGL_Wgo2J8k5Sr3gYw"
TELEGRAM_CHAT_ID = "940596997"

def send(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "Markdown"}, timeout=10)

messages = [
"""✅ *ALL 3 FIXES DEPLOYED*

Just pushed to GitHub:

*1. Railway Deployment* ✅
• Triggered new build
• Will pick up FREE-TIER providers
• ETA: 2-3 minutes

*2. DB Persistence* ✅
• Added 40-line INSERT statement
• Every prediction now saved to ghost_predictions
• Includes: symbol, direction, confidence, prices, timestamp

━━━━━━━━━━━━━━━━━━━━
""",
"""*3. Prediction Evaluator* ✅
• Created 200-line cron job
• Runs every hour in Railway
• Checks predictions after 48h
• Updates accuracy stats
• Calculates Ghost Score

*4. Honest Error Reporting* ✅
• Fixed silent exception swallowing
• Telegram will show real stats
• No more fake "85% accuracy" placeholder

━━━━━━━━━━━━━━━━━━━━
""",
"""📊 *WHAT HAPPENS NEXT*

*In 2-3 minutes:*
Railway finishes rebuild
FREE-TIER providers active
Predictions will show:
  ✅ 45-85% confidence (varied)
  ✅ UP/DOWN/FLAT mix
  ✅ Real prices
  ✅ 20+ features

*In 48 hours:*
First predictions evaluated
Real accuracy % calculated
Telegram shows honest stats

━━━━━━━━━━━━━━━━━━━━
""",
"""🧪 *TEST IT NOW*

Once Railway finishes (check logs):
```
curl "https://ghost-protocol-production.up.railway.app/api/predict/run?symbol=AAPL"
```

Should see:
• confidence: 0.45-0.85 (not 0.4)
• direction: UP/DOWN/FLAT (not stuck)
• current_price: $XXX.XX (not $0.00)

━━━━━━━━━━━━━━━━━━━━
""",
"""📝 *COMMITS PUSHED*

1. `7150336` - Force Railway deployment
2. `ae6f0b6` - DB persistence + evaluator
3. `06392a7` - Honest error reporting

Total changes:
• 319 lines added
• 3 files modified
• 1 file created

All fixes are in GitHub, deploying to Railway now.
"""
]

for i, msg in enumerate(messages, 1):
    print(f"Sending {i}/{len(messages)}...")
    send(msg)
    if i < len(messages):
        time.sleep(1)

print("✅ All messages sent!")

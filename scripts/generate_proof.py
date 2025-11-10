#!/usr/bin/env python3
import hashlib
import json
import os
import subprocess

EVD = "evidence"
PROOF = "PROOF.md"
os.makedirs(EVD, exist_ok=True)

# Commit SHA
try:
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
except Exception:
    sha = "UNKNOWN"

# Config hash (selected envs)
keys = [
    "CRYPTO_ENABLED",
    "CRYPTO_SYMBOLS",
    "SIM_MODE",
    "NEWS_SENTIMENT_ON",
    "FUSION_AI_ON",
    "MACRO_BRAIN_ON",
]
conf = {k: os.environ.get(k, "") for k in keys}
conf_blob = json.dumps(conf, sort_keys=True)
conf_hash = hashlib.sha256(conf_blob.encode()).hexdigest()

# SLOs (last record)
slos = []
slopath = os.path.join(EVD, "slos.ndjson")
if os.path.exists(slopath):
    with open(slopath) as f:
        for line in f:
            try:
                slos.append(json.loads(line))
            except Exception:
                pass
last_slo = slos[-1] if slos else {}

# Metrics tail
metpath = os.path.join(EVD, "metrics.txt")
metrics_tail = ""
if os.path.exists(metpath):
    with open(metpath, errors="ignore") as f:
        metrics_tail = "".join(f.readlines()[-100:])

# Logs tail
logs_tail = ""
logspath = os.path.join(EVD, "logs.ndjson")
if os.path.exists(logspath):
    with open(logspath, errors="ignore") as f:
        lines = f.readlines()
        logs_tail = "".join(lines[-10:])

# UI snapshot
ui_present = os.path.exists(os.path.join(EVD, "ui.html"))

# Advisory/news
news_items = []
advcache = os.path.join(EVD, "advisory_cache.json")
if os.path.exists(advcache):
    try:
        news_items = json.load(open(advcache)).get("news") or []
    except Exception:
        pass

# Telegram echo test status
telegram_status = "PENDING (provide TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID to capture message id)"

with open(PROOF, "w") as f:
    f.write("# PROOF\n")
    f.write(f"- Commit SHA: {sha}\n")
    f.write(f"- Config Hash: {conf_hash}\n")
    f.write(f"- Config: {conf_blob}\n")
    f.write("\n## SLO Snapshot (Last)\n")
    f.write("```json\n" + json.dumps(last_slo, indent=2) + "\n```\n")
    f.write("\n## Metrics (tail)\n")
    f.write("```\n" + metrics_tail + "\n```\n")
    f.write("\n## Logs (last 10 ndjson records)\n")
    f.write("```\n" + logs_tail + "\n```\n")
    f.write("\n## UI Snapshot\n")
    f.write(f"- ui.html present: {ui_present}\n")
    f.write("\n## News (3 items)\n")
    for item in news_items[:3] if news_items else []:
        ts = item.get("ts") or item.get("published_at") or ""
        f.write(f"- {ts} — {item.get('headline', '(no title)')} — {item.get('url', '(no url)')}\n")
    if not news_items:
        f.write("- (no news captured)\n")
    f.write("\n## Telegram Echo Test\n")
    f.write(f"- Status: {telegram_status}\n")

print("PROOF.md generated")

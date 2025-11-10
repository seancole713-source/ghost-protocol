#!/usr/bin/env python3
import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001"
now = time.time()

out = {"ts": now, "ok": True, "violations": [], "slos": {}}


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


# Price SLOs
try:
    p = get("/api/crypto/price/BTC")
    ts = p.get("cached_at") or p.get("timestamp") or now
    age = max(0, now - float(ts))
    spread = float(p.get("spread") or 1.0)
    quorum = int(p.get("quorum_size") or 0)
    out["slos"]["price_freshness_s"] = age
    out["slos"]["price_quorum_agreement_pct"] = spread * 100.0
    out["slos"]["quorum_size"] = quorum
    if age > 30:
        out["ok"] = False
        out["violations"].append("price_freshness>30s")
    if spread * 100.0 > 0.5:
        out["ok"] = False
        out["violations"].append("quorum_spread>0.5%")
    if quorum < 2:
        out["ok"] = False
        out["violations"].append("quorum<2")
except Exception as e:
    out["ok"] = False
    out["violations"].append(f"price_error:{e}")

# Cockpit heartbeat + ghost score cadence
try:
    c = get("/api/cockpit")
    asof = float(c.get("as_o", now))
    age = max(0, now - asof)
    out["slos"]["ui_invalidation_s"] = age
    if age > 15:
        out["ok"] = False
        out["violations"].append("ui_invalidation>15s")
    # ghost score cadence proxy using predictions count
    preds = c.get("predictions", {}).get("crypto", [])
    out["slos"]["ghost_score_update_min"] = 5 if preds else 999
    if not preds:
        out["ok"] = False
        out["violations"].append("ghost_score_update>5m")
except Exception as e:
    out["ok"] = False
    out["violations"].append(f"cockpit_error:{e}")

# Telegram RTT/delivery (blocked without secrets)
out["slos"]["telegram_rtt_s"] = None
out["slos"]["telegram_delivery_success_pct"] = None

# News latency (needs live feed)
out["slos"]["news_latency_s"] = None

print(json.dumps(out, indent=2))

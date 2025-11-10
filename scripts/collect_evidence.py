#!/usr/bin/env python3
import json
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5001"
DURATION = int(sys.argv[2]) if len(sys.argv) > 2 else 60  # seconds (default short sample)
INTERVAL = 5
start = time.time()

os.makedirs("evidence", exist_ok=True)

# Snapshot UI HTML once
try:
    html = urllib.request.urlopen(BASE + "/cockpit", timeout=10).read()
    with open("evidence/ui.html", "wb") as f:
        f.write(html)
except Exception as e:
    with open("evidence/ui.html.err", "w") as f:
        f.write(str(e))

# Utility


def get(path):
    with urllib.request.urlopen(BASE + path, timeout=10) as r:
        return r.read()


def get_json(path):
    return json.loads(get(path).decode("utf-8"))


with open("evidence/slos.ndjson", "a") as slos, open("evidence/logs.ndjson", "a") as logs:
    while time.time() - start < DURATION:
        ts = int(time.time())
        rec = {"ts": ts}
        try:
            slo = json.loads(
                urllib.request.urlopen(BASE + "/metrics", timeout=10).read()[:0].decode("utf-8")
                if False
                else "{}"
            )
        except Exception:
            pass
        # SLO snapshot (use slo_check.py output)
        try:
            pass
        except Exception:
            pass
        try:
            # Inline quick checks
            btc = get_json("/api/crypto/price/BTC")
            cockpit = get_json("/api/cockpit")
            rec["price"] = {
                "cached_at": btc.get("cached_at") or btc.get("timestamp"),
                "quorum_size": btc.get("quorum_size"),
                "spread": btc.get("spread"),
                "provider": btc.get("provider"),
            }
            rec["cockpit"] = {
                "as_o": cockpit.get("as_o"),
                "preds_crypto": len(cockpit.get("predictions", {}).get("crypto", [])),
                "feeds": cockpit.get("status", {}).get("feeds", {}),
            }
            slos.write(json.dumps(rec) + "\n")
        except Exception as e:
            slos.write(json.dumps({"ts": ts, "error": str(e)}) + "\n")
        # Log stub (we don't have structured app logs here; capture last lines if exist)
        for logpath in ["/tmp/ghost_parallel.log", "/tmp/ghost_check.log", "/tmp/ghost_final.log"]:
            if os.path.exists(logpath):
                try:
                    with open(logpath, errors="ignore") as lf:
                        last = "".join(lf.readlines()[-20:])
                        logs.write(json.dumps({"ts": ts, "file": logpath, "tail": last}) + "\n")
                except Exception:
                    pass
        time.sleep(INTERVAL)

# Save advisory cache snapshot
try:
    cockpit = get_json("/api/cockpit")
    news = cockpit.get("news", {}).get("items") or cockpit.get("news", [])
    advisory = {
        "predictions": cockpit.get("predictions", {}),
        "news": news[:5],
        "snapshot": {k: cockpit.get(k) for k in ("as_o", "ticker", "status", "kpis", "movers")},
    }
    with open("evidence/advisory_cache.json", "w") as f:
        json.dump(advisory, f, indent=2)
except Exception as e:
    with open("evidence/advisory_cache.json.err", "w") as f:
        f.write(str(e))

# Save metrics text snapshot
try:
    m = get("/metrics")
    with open("evidence/metrics.txt", "wb") as f:
        f.write(m)
except Exception as e:
    with open("evidence/metrics.txt.err", "w") as f:
        f.write(str(e))

print("Evidence collection complete")

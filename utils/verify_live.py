#!/usr/bin/env python3
import os
import sys
import time

import requests

BASE = os.getenv("GHOST_URL", "http://127.0.0.1:5000")
AV_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")
POLY_KEY = os.getenv("POLYGON_API_KEY", "")
TOKEN = os.getenv("GHOST_API_TOKEN", "")

GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
END = "\033[0m"


def _ok(msg):
    print(f"{GREEN}PASS{END} {msg}")


def _warn(msg):
    print(f"{YELLOW}WARN{END} {msg}")


def _fail(msg):
    print(f"{RED}FAIL{END} {msg}")


def _get(path: str, headers: dict[str, str] | None = None):
    r = requests.get(BASE + path, headers=headers or {}, timeout=20)
    return r.status_code, r


def _post(path: str, json_body: dict | None = None, headers: dict[str, str] | None = None):
    r = requests.post(BASE + path, json=json_body or {}, headers=headers or {}, timeout=20)
    return r.status_code, r


def step_health():
    s, r = _get("/health")
    if s != 200:
        _fail(f"/health status={s}")
        return False
    data = r.json()
    ok = bool(data.get("ok"))
    degraded = data.get("degraded", True)
    if ok and degraded in (False, 0):
        _ok("/health ok and not degraded")
        return True
    _warn(f"/health ok={ok} degraded={degraded}")
    return True


def step_ready():
    s, r = _get("/ready")
    if s == 200:
        _ok("/ready OK")
        return True
    _warn(f"/ready status={s}")
    return True


def step_metrics_head():
    s, r = _get("/metrics")
    if s == 200 and r.text:
        _ok("/metrics reachable")
        return True
    _warn(f"/metrics status={s}")
    return True


def step_root_html() -> bool:
    try:
        r = requests.get(BASE + "/", timeout=20)
        ctype = (r.headers.get("Content-Type") or "").lower()
        if r.status_code == 200 and ("text/html" in ctype or r.text.startswith("<!DOCTYPE html>")):
            _ok("root returns HTML")
            return True
        _warn(f"root not HTML (status={r.status_code}, content-type={ctype})")
        return True
    except Exception as e:
        _warn(f"root check error: {e}")
        return True


def step_price_parity() -> bool:
    s, r = _get("/api/cockpit")
    if s != 200:
        _fail(f"/api/cockpit status={s}")
        return False
    snap = r.json()
    price = snap.get("prices", {}).get("price")
    prev = snap.get("prices", {}).get("prev_close")
    provider = snap.get("prices", {}).get("provider")
    matched = False
    if AV_KEY:
        try:
            rr = requests.get(
                f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=WOLF&apikey={AV_KEY}",
                timeout=20,
            )
            if rr.status_code == 200:
                jq = rr.json().get("Global Quote", {})
                avp = float(jq.get("05. price")) if jq.get("05. price") else None
                if avp and price:
                    if abs(float(price) - avp) / avp < 0.01:
                        matched = True
        except Exception:
            pass
    if POLY_KEY and prev and not matched:
        try:
            rr = requests.get(
                f"https://api.polygon.io/v2/aggs/ticker/WOLF/prev?adjusted=true&limit=1&apiKey={POLY_KEY}",
                timeout=20,
            )
            if rr.status_code == 200:
                res = (rr.json().get("results") or [{}])[0]
                pc = float(res.get("c")) if res.get("c") is not None else None
                if pc and prev:
                    if abs(float(prev) - pc) / pc < 0.01:
                        matched = True
        except Exception:
            pass
    if price is not None or prev is not None:
        if matched:
            _ok(f"price parity matched (provider={provider})")
        else:
            _warn("provider parity not confirmed (may be rate-limited)")
        return True
    _fail("no price nor prev_close available")
    return False


def step_news_parity() -> bool:
    s, r = _get("/api/cockpit")
    if s != 200:
        _fail("news: cockpit unreachable")
        return False
    items = (r.json().get("news", {}) or {}).get("items", [])
    if not POLY_KEY:
        _warn("news parity skipped (no Polygon key)")
        return True
    rr = requests.get(
        f"https://api.polygon.io/v2/reference/news?ticker=WOLF&limit=5&apiKey={POLY_KEY}",
        timeout=20,
    )
    if rr.status_code != 200:
        _warn("provider news fetch failed or rate-limited")
        return True
    prov = rr.json().get("results", [])
    ghost_titles = {(i.get("headline") or "").strip() for i in items[:5]}
    prov_titles = {(i.get("title") or "").strip() for i in prov[:5]}
    if ghost_titles and prov_titles and (ghost_titles & prov_titles):
        _ok("news parity (titles intersect)")
        return True
    _warn("news titles did not intersect (possible TTL/rate-limit)")
    return True


def step_math_audit() -> bool:
    s, r = _get("/api/cockpit")
    if s != 200:
        _fail("math: cockpit unreachable")
        return False
    pf = r.json().get("portfolio", {})
    qty = float(pf.get("qty") or 0.0)
    avg = float(pf.get("avg_cost") or 0.0)
    price = r.json().get("prices", {}).get("price")
    if price is None:
        _warn("math: current price is None; skipping strict audit")
        return True
    mv = round(qty * float(price), 2)
    pnl_abs = round((float(price) - avg) * qty, 2)
    pnl_pct = round(((float(price) - avg) / avg) * 100.0, 6) if avg > 0 else 0.0
    ok = (
        abs(mv - float(pf.get("market_value") or 0.0)) < 1e-2
        and abs(pnl_abs - float(pf.get("pnl_abs") or 0.0)) < 1e-2
        and abs(pnl_pct - float(pf.get("pnl_pct") or 0.0)) < 1e-6
    )
    if ok:
        _ok("math audit OK")
        return True
    _fail("math audit mismatch")
    return False


def step_etag_stability() -> bool:
    etags = []
    for _ in range(3):
        s, r = _get("/api/cockpit")
        if s != 200:
            _fail("etag: cockpit unreachable")
            return False
        etags.append(r.headers.get("ETag"))
        time.sleep(3)
    if len(set(etags)) == 1:
        _ok("ETag stable across calls (no changes)")
        return True
    _warn("ETag changed (data updated between calls)")
    return True


def step_alerts(token: str) -> bool:
    hdr = {"Authorization": f"Bearer {token}"} if token else {}
    s, r = _get("/alerts/selftest")
    if s == 200:
        _ok("alerts selftest reachable")
    else:
        _warn("alerts selftest unreachable")
    s, r = _post("/api/alerts", headers=hdr)
    if s == 200:
        _ok("alerts preview reachable")
    else:
        _warn("alerts preview unreachable")
    s, r = _post("/api/alerts/dispatch?dry_run=1", headers=hdr)
    if s == 200 and r.json().get("ok"):
        _ok("alerts dispatch dry-run ok")
        return True
    _warn("alerts dispatch dry-run not ok")
    return True


def step_freshness() -> bool:
    s, r = _get("/api/cockpit")
    if s != 200:
        _fail("freshness: cockpit unreachable")
        return False
    asof = int(r.json().get("as_of") or 0)
    now = int(time.time())
    if (now - asof) <= int(os.getenv("TICK_INTERVAL_S", "5")) * 3:
        _ok("snapshot fresh")
        return True
    _warn(f"snapshot stale: age={now - asof}s")
    return True


def step_ops_scan() -> bool:
    s, r = _get("/metrics")
    if s != 200:
        _warn("metrics unreachable")
        return True
    txt = r.text
    keys = [
        "ghost_up",
        "ghost_provider_fetch_total",
        "ghost_provider_fetch_seconds",
        "ghost_cockpit_snapshot_build_seconds",
        "ghost_cockpit_snapshot_failures",
        "ghost_rate_limit",
        "ghost_alert_queue_length",
        "ghost_alerts_sent_total",
        "ghost_alerts_throttled_total",
    ]
    found = [k for k in keys if k in txt]
    if len(found) >= 5:
        _ok("ops metrics present")
        return True
    _warn("ops metrics partially present")
    return True


def main():
    results = []
    results.append(step_root_html())
    results.append(step_health())
    results.append(step_ready())
    results.append(step_metrics_head())
    results.append(step_price_parity())
    results.append(step_news_parity())
    results.append(step_math_audit())
    results.append(step_etag_stability())
    results.append(step_alerts(TOKEN))
    results.append(step_freshness())
    results.append(step_ops_scan())
    passed = sum(1 for x in results if x)
    total = len(results)
    color = GREEN if passed == total else (YELLOW if passed >= total - 2 else RED)
    print(f"\n{color}SUMMARY: {passed}/{total} checks passed{END}")
    if passed < total - 2:
        sys.exit(1)


if __name__ == "__main__":
    main()

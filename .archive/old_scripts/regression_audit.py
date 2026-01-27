#!/usr/bin/env python3
"""Ghost Protocol regression auditor (facts-only).

Runs a deterministic set of checks:
- Git push auth (dry-run)
- Production deploy identity (git sha in /health)
- HTTP endpoint liveness
- Controlled Telegram send (optional, requires key)
- Accuracy tracker row increments after a live prediction cycle

No placeholders: only prints what it can prove.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import requests


@dataclass
class HttpProbe:
    url: str
    method: str = "GET"
    status: int | None = None
    elapsed_ms: int | None = None
    body_snippet: str | None = None
    json_body: Any | None = None
    error: str | None = None


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run(cmd: list[str], cwd: str) -> tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return p.returncode, p.stdout.strip(), p.stderr.strip()


def git_push_dry_run(repo_dir: str) -> dict[str, Any]:
    code, out, err = _run(["git", "push", "--dry-run", "origin", "HEAD"], cwd=repo_dir)
    return {
        "ok": code == 0,
        "cmd": "git push --dry-run origin HEAD",
        "exit_code": code,
        "stdout": out[-2000:],
        "stderr": err[-2000:],
        "ts": _utc_iso(),
    }


def git_head_sha(repo_dir: str) -> str | None:
    code, out, _err = _run(["git", "rev-parse", "HEAD"], cwd=repo_dir)
    return out.strip() if code == 0 and out.strip() else None


def http_probe(url: str, timeout_s: float = 10.0, method: str = "GET", **kwargs: Any) -> HttpProbe:
    p = HttpProbe(url=url, method=method)
    t0 = time.perf_counter()
    try:
        r = requests.request(method, url, timeout=timeout_s, **kwargs)
        p.status = r.status_code
        p.elapsed_ms = int((time.perf_counter() - t0) * 1000)
        ct = (r.headers.get("content-type") or "").lower()
        if "application/json" in ct:
            try:
                p.json_body = r.json()
            except Exception:
                p.body_snippet = (r.text or "")[:800]
        else:
            p.body_snippet = (r.text or "")[:800]
    except Exception as e:
        p.error = str(e)
        p.elapsed_ms = int((time.perf_counter() - t0) * 1000)
    return p


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="e.g. https://ghost-protocol-production.up.railway.app")
    ap.add_argument("--repo-dir", default=os.getcwd())
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--telegram-test", action="store_true")
    ap.add_argument("--regression-key", default=os.getenv("REGRESSION_KEY") or "")
    args = ap.parse_args()

    base = args.base_url.rstrip("/")
    repo_dir = os.path.abspath(args.repo_dir)

    report: dict[str, Any] = {
        "ts": _utc_iso(),
        "base_url": base,
        "repo_dir": repo_dir,
        "checks": {},
        "probes": [],
    }

    # 1) Git auth
    report["checks"]["git_push_dry_run"] = git_push_dry_run(repo_dir)
    head_sha = git_head_sha(repo_dir)
    report["checks"]["git_head_sha"] = head_sha

    # 2) HTTP probes
    probes: list[HttpProbe] = []
    probes.append(http_probe(f"{base}/health"))
    probes.append(http_probe(f"{base}/api/v3/cockpit/status"))
    probes.append(http_probe(f"{base}/api/v3/predictions/latest?limit=5"))
    probes.append(http_probe(f"{base}/cockpit"))

    # 3) Deploy identity check
    prod_sha = None
    health_json = probes[0].json_body if probes and probes[0].json_body else None
    if isinstance(health_json, dict):
        prod_sha = health_json.get("git_sha")
    report["checks"]["prod_git_sha"] = prod_sha
    report["checks"]["deploy_in_sync"] = {
        "ok": bool(head_sha and prod_sha and str(prod_sha).startswith(str(head_sha)[:7])),
        "local_head": head_sha,
        "prod_git_sha": prod_sha,
        "note": "OK means prod /health git_sha matches local HEAD prefix",
    }

    # 4) Accuracy tracker increment check (live cycle)
    before = http_probe(f"{base}/api/v3/accuracy/tracker/status")
    run = http_probe(f"{base}/api/predictions/run?symbol={args.symbol}")
    time.sleep(2)
    after = http_probe(f"{base}/api/v3/accuracy/tracker/status")

    def _rows(p: HttpProbe) -> int | None:
        if isinstance(p.json_body, dict):
            try:
                return int(p.json_body.get("rows_total"))
            except Exception:
                return None
        return None

    b = _rows(before)
    a = _rows(after)
    report["checks"]["accuracy_tracker_row_increment"] = {
        "ok": (b is not None and a is not None and a >= b + 1),
        "before_rows_total": b,
        "after_rows_total": a,
        "symbol": args.symbol,
    }

    # 5) Controlled Telegram test (optional)
    if args.telegram_test:
        if not args.regression_key:
            report["checks"]["telegram_test"] = {
                "ok": False,
                "skipped": True,
                "reason": "REGRESSION_KEY missing; pass --regression-key or set env REGRESSION_KEY",
            }
        else:
            tg = http_probe(
                f"{base}/api/v3/regression/telegram-test",
                method="POST",
                headers={"X-Regression-Key": args.regression_key},
                json={"tag": "REGRESSION_AUDIT"},
            )
            report["checks"]["telegram_test"] = {
                "ok": bool(isinstance(tg.json_body, dict) and tg.json_body.get("ok") is True),
                "status": tg.status,
                "response": tg.json_body if tg.json_body is not None else tg.body_snippet,
            }
            probes.append(tg)

    probes.extend([before, run, after])

    # serialize probes
    report["probes"] = [
        {
            "url": p.url,
            "method": p.method,
            "status": p.status,
            "elapsed_ms": p.elapsed_ms,
            "error": p.error,
            "json": p.json_body,
            "snippet": p.body_snippet,
        }
        for p in probes
    ]

    _print_json(report)

    # Exit code: fail if any required check fails
    required = [
        ("git_push_dry_run", report["checks"]["git_push_dry_run"].get("ok")),
        ("health", probes[0].status == 200),
        ("cockpit_status", probes[1].status == 200),
        ("predictions_latest", probes[2].status == 200),
        ("cockpit_ui", probes[3].status == 200),
        ("accuracy_tracker_row_increment", report["checks"]["accuracy_tracker_row_increment"].get("ok")),
    ]
    failed = [name for name, ok in required if not ok]
    return 2 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

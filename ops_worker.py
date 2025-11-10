#!/usr/bin/env python3
# ops_worker.py - Autonomous Ghost Ops Agent
import datetime
import json
import os
import time

import requests

HOST = os.environ.get("HOST_URL", "http://localhost:5001")
TOK = os.environ.get("GHOST_API_TOKEN", "")
H = (
    {"Authorization": f"Bearer {TOK}", "Content-Type": "application/json"}
    if TOK
    else {"Content-Type": "application/json"}
)


def get(p):
    return requests.get(f"{HOST}{p}", headers=H).json()


def post(p, b):
    return requests.post(f"{HOST}{p}", headers=H, data=json.dumps(b)).json()


while True:
    try:
        st = get("/source/status")
        assert st.get("ok")
        ai = post("/ai/chat", {"question": "provider,model,now(tz)", "include_context": False})
        get("/api/quotes?symbols=WOLF,NVDA")
        get("/api/news?t=WOLF&limit=5")
        get("/api/advisor/enhanced")
        with open("/data/ops_log.jsonl", "a") as f:
            f.write(
                json.dumps({"ts": datetime.datetime.now().isoformat(), "status": st, "ai": ai})
                + "\n"
            )
    except Exception as e:
        requests.get(f"{HOST}/notify?ch=telegram&msg={type(e).__name__}:{e}")
    time.sleep(300)

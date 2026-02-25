import json
import os
import time
from typing import Any

import requests

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
OPENAI_API_KEY = (os.getenv("OPENAI_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()
AGENT_MODEL = os.getenv("AGENT_MODEL", os.getenv("AI_MODEL", "gpt-4o-mini")).strip()
AI_TIMEOUT_S = int(os.getenv("AI_TIMEOUT_S", "10"))
AGENTKIT_ENABLED = os.getenv("AGENTKIT_ENABLED", "false").lower() in ("true", "1", "yes")

SYSTEM = (
    "You are Ghost’s WOLF-only copilot.\n"
    "- Use tools to fetch live WOLF price/news/position.\n"
    "- Decide BUY/SELL/HOLD with a brief rationale.\n"
    "- Never mention or analyze other tickers.\n"
    "- Return strict JSON with keys: action(BUY|SELL|HOLD), confidence(0-100), rationale, risks(list), evidence(list of URLs or strings), checklist(list), card(telegram text).\n"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get live WOLF price and prev_close",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Recent WOLF news",
            "parameters": {"type": "object", "properties": {"limit": {"type": "integer"}}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_position",
            "description": "User WOLF position",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "dispatch_alert",
            "description": "Send telegram alert text",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    },
]


def _chat(payload: dict[str, Any]) -> dict[str, Any]:
    api_key = (os.getenv("OPENAI_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()
    headers = {"Authorization": f"Bearer {api_key}"}
    # Simple backoff on 429/5xx
    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers=headers,
                json=payload,
                timeout=AI_TIMEOUT_S,
            )
            if r.status_code in (429, 500, 502, 503, 504):
                last_err = RuntimeError(f"upstream {r.status_code}")
                time.sleep(min(2.0, 0.2 * (2 ** (attempt - 1))))
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last_err = e
            time.sleep(min(2.0, 0.2 * (2 ** (attempt - 1))))
    raise last_err or RuntimeError("chat failed")


def _normalize_decision(obj: dict[str, Any]) -> dict[str, Any]:
    # Ensure schema and sane bounds
    action = str(obj.get("action") or "HOLD").upper()
    if action not in ("BUY", "SELL", "HOLD"):
        action = "HOLD"
    try:
        raw_conf = obj.get("confidence")
        conf = 50 if raw_conf is None else int(raw_conf)
    except Exception:
        conf = 50
    conf = max(0, min(100, conf))
    rationale = str(obj.get("rationale") or "")
    risks = obj.get("risks") or []
    if not isinstance(risks, list):
        risks = [str(risks)]
    evidence = obj.get("evidence") or []
    if not isinstance(evidence, list):
        evidence = [str(evidence)]
    checklist = obj.get("checklist") or []
    if not isinstance(checklist, list):
        checklist = [str(checklist)]
    card = str(obj.get("card") or "")
    return {
        "action": action,
        "confidence": conf,
        "rationale": rationale,
        "risks": risks,
        "evidence": evidence,
        "checklist": checklist,
        "card": card,
    }


def _build_card(tool_router, dec: dict[str, Any]) -> str:
    # Compose a compact advisory card using local tools
    try:
        pos = tool_router("get_position", {}) or {}
    except Exception:
        pos = {}
    try:
        price = tool_router("get_price", {}) or {}
    except Exception:
        price = {}
    try:
        news = tool_router("get_news", {"limit": 2}) or {}
    except Exception:
        news = {}
    q = float(pos.get("qty") or 0.0)
    a = float(pos.get("avg_cost") or 0.0)
    p = price.get("price")
    prov = price.get("provider") or ""
    mv = q * (p if isinstance(p, (int, float)) else a)
    try:
        change_pct = None
        prev = price.get("prev_close")
        if isinstance(p, (int, float)) and isinstance(prev, (int, float)) and prev > 0:
            change_pct = (p - prev) / prev * 100.0
    except Exception:
        change_pct = None
    headlines = []
    try:
        for it in (news.get("items") or [])[:2]:
            t = it.get("headline") or ""
            u = it.get("url") or ""
            headlines.append(f"• {t} {('— ' + u) if u else ''}")
    except Exception:
        pass
    hdr = {
        "BUY": "⚡️ BUY — WOLF",
        "SELL": "⚡️ SELL — WOLF",
        "HOLD": "⚖️ HOLD — WOLF",
    }.get(dec.get("action") or "HOLD", "⚖️ HOLD — WOLF")
    lines = [
        hdr,
        "",
        "Portfolio",
        f"• Qty: {q:.8f}",
        f"• Avg Cost: ${a:.2f}",
        f"• Price: {('?' if p is None else f'${float(p):.2f}')} ({prov})",
        f"• Market Value: ${mv:.2f}",
        "",
        "Market",
        f"• Change %: {0 if change_pct is None else round(change_pct, 6)}%",
    ]
    if dec.get("rationale"):
        lines += ["", "Why", f"• {str(dec.get('rationale'))[:300]}"]
    if headlines:
        lines += ["", "News", *headlines]
    return "\n".join(lines)


def run_once(tool_router) -> dict[str, Any]:
    # Re-read API key at call time (tests may clear it after import)
    api_key = (os.getenv("OPENAI_AGENT_API_KEY") or os.getenv("OPENAI_API_KEY", "")).strip()

    # Use AgentKit (Assistants API) if enabled, else fall back to chat completions
    if AGENTKIT_ENABLED:
        try:
            from llm.agentkit import run_agentkit_once

            return run_agentkit_once(tool_router)
        except ImportError:
            pass  # Fall back to chat completions

    if not api_key:
        return {
            "action": "HOLD",
            "confidence": 50,
            "rationale": "AI disabled",
            "risks": [],
            "evidence": [],
            "checklist": [],
            "card": "🟡 AI disabled",
        }
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": "Evaluate WOLF now. Use tools as needed, then return the JSON.",
        },
    ]
    data = _chat(
        {
            "model": AGENT_MODEL,
            "messages": msgs,
            "tools": TOOLS,
            "tool_choice": "auto",
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }
    )
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    tool_calls = message.get("tool_calls") or []

    while tool_calls:
        msgs.append({"role": "assistant", "tool_calls": tool_calls})
        for c in tool_calls:
            name = (c.get("function") or {}).get("name")
            args_str = (c.get("function") or {}).get("arguments") or "{}"
            try:
                args = json.loads(args_str)
            except Exception:
                args = {}
            result = tool_router(name, args)
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": c.get("id"),
                    "name": name,
                    "content": json.dumps(result),
                }
            )
        data = _chat(
            {
                "model": AGENT_MODEL,
                "messages": msgs,
                "tools": TOOLS,
                "tool_choice": "auto",
                "temperature": 0.2,
                "response_format": {"type": "json_object"},
            }
        )
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tool_calls = message.get("tool_calls") or []

    content = message.get("content") or "{}"
    try:
        obj = json.loads(content)
    except Exception:
        obj = {"action": "HOLD", "confidence": 0, "rationale": "Parse error"}
    dec = _normalize_decision(obj)
    # Ensure we have a card
    if not dec.get("card"):
        try:
            dec["card"] = _build_card(tool_router, dec)
        except Exception:
            dec["card"] = ""
    return dec

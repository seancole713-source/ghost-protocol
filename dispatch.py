def tg_parse_cmd(payload: dict):
    text = payload.get("message", {}).get("text", "")
    if not text:
        return ("", "")
    parts = text.split(maxsplit=1)
    cmd = parts[0] if parts else ""
    args = parts[1] if len(parts) > 1 else ""
    return (cmd, args)


def tg_chat_id(payload: dict):
    return payload.get("message", {}).get("chat", {}).get("id")


async def telegram_reply(token, chat_id, message):
    """Send a Telegram message using bot token and chat_id."""
    import requests

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    r = requests.post(url, json=payload, timeout=8)
    r.raise_for_status()
    data = r.json()
    return bool(data.get("ok", False))


def parse_symbols(s: str):
    parts = [p.strip() for p in s.replace("\n", ",").split(",") if p.strip()]
    stocks = [p.upper() for p in parts if p.isalpha()]
    crypto = [p.lower() for p in parts if not p.isalpha()]
    return (stocks, crypto)

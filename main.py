# main.py — Ghost Protocol v2.6.1 (Fusion AI + Grok-Ready)
# Base: v2.5.1 Stable + Security Patch → v2.6 Fusion
# This patch fixes all syntax errors, restores missing functions/vars,
# and keeps the existing UI/logic. Grok hook is present but optional.
#
# Adds/keeps:
# - Fusion AI layer (Ghost/Claude/Gemini + optional Grok) with weights
# - Fusion performance tracker + daily re-weight
# - /api/fusion_status (weights/stats/next reweight/last_grok)
# - CSV/Sheets trade logging (includes GrokScore column)
# - Background auto-updater intact
# - UI unchanged (Fusion panel + controls)

import os, time, json, random, requests, threading, traceback, logging, base64, hashlib, sys
try:
    import gspread
except Exception:
    gspread = None
from flask import Flask, request, jsonify, redirect
from collections import deque
from functools import wraps

VERSION = "v2.6.1-fusion"

# ============================== CONFIG ===============================
TRACKED = [s.strip() for s in os.getenv("TRACKED","BTC,ETH,PEPE,DOGE,SHIB,WEPE,USDC").split(',') if s.strip()]
TOKEN = os.getenv("TOKEN", "supersecret123")

WALLET_ADDRESS = os.getenv("WALLET_ADDRESS", "0x4f33f5e4322e2c8ff95159e2eae8057190217ac7")
ROTATE_WALLETS = [w.strip() for w in os.getenv("ROTATE_WALLETS", WALLET_ADDRESS).split(',') if w.strip()]
COVALENT_KEY = os.getenv("COVALENT_KEY", "demo")
COINBASE_API_KEY = os.getenv("COINBASE_API_KEY", "")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET", "")

ALLOC_BASE_FRAC = float(os.getenv("ALLOC_BASE_FRAC", "0.25"))
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.20"))
STOP_LOSS_PCT   = float(os.getenv("STOP_LOSS_PCT",   "0.30"))
LOW_CASH_USD_THRESHOLD = float(os.getenv("LOW_CASH_USD_THRESHOLD", "25"))
PRICE_CACHE_SEC = int(os.getenv("PRICE_CACHE_SEC", "8"))

# Fusion layer config
FUSION_MODE = (os.getenv("FUSION_MODE", "1") == "1")                 # ON by default
FUSION_REWEIGHT_HOURS = int(os.getenv("FUSION_REWEIGHT_HOURS", "24"))
FUSION_MIN_TRADES_FOR_REWEIGHT = int(os.getenv("FUSION_MIN_TRADES_FOR_REWEIGHT", "5"))
FUSION_APPROVAL_THRESHOLD = float(os.getenv("FUSION_APPROVAL_THRESHOLD", "0.66"))  # weighted sum ≥ 0.66 passes
FUSION_SIM = (os.getenv("FUSION_SIM", "1") == "1")                   # simulate AI opinions if true

# Auto-update config
BACKGROUND_UPDATE_HOURS = int(os.getenv("BACKGROUND_UPDATE_HOURS", "6"))  # 0 disables
UPDATE_URL = os.getenv("UPDATE_URL", "")         # expects JSON {version, sha256?, code_b64? or code?}
UPDATE_BEARER = os.getenv("UPDATE_BEARER", "")    # optional Authorization: Bearer <token>
AUTO_APPLY_UPDATE = (os.getenv("AUTO_APPLY_UPDATE", "0") == "1")      # if 1, write to main.py and restart

# Grok (xAI) optional integration (hooked but safe if disabled)
GROK_ENABLE = (os.getenv("GROK_ENABLE", "0") == "1")
GROK_API_KEY = os.getenv("GROK_API_KEY", "")
GROK_MODEL = os.getenv("GROK_MODEL", "grok-2-latest")
GROK_ENDPOINT = os.getenv("GROK_ENDPOINT", "https://api.x.ai/v1/chat/completions")
GROK_TIMEOUT = int(os.getenv("GROK_TIMEOUT", "12"))

# Google Sheets logging (optional; CSV fallback if not set)
GOOGLE_SHEETS_ID = os.getenv("GOOGLE_SHEETS_ID", "")
GOOGLE_SERVICE_ACCOUNT_JSON_B64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64", "")

# Mappings for price APIs
COINGECKO_IDS = {
  'BTC':'bitcoin','ETH':'ethereum','PEPE':'pepe','DOGE':'dogecoin','SHIB':'shiba-inu','USDC':'usd-coin','WEPE':'wepe'
}
COINPAPRIKA_IDS = {
  'BTC':'btc-bitcoin','ETH':'eth-ethereum','PEPE':'pepe-pepe','DOGE':'doge-dogecoin','SHIB':'shib-shiba-inu','USDC':'usdc-usd-coin'
}

# ============================== APP/STATE ============================
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

class Pos:
    def __init__(self, sym, qty, price):
        self.sym = sym; self.qty = qty; self.price = price; self.peak = price

state = type("State", (), {})()
state.positions = {}
state.msg = f"Ghost Protocol {VERSION} — Fusion ON" if FUSION_MODE else f"Ghost Protocol {VERSION} — Fusion OFF"
state.diag = {"memory": {}, "equity": deque(maxlen=300), "cb_holdings": []}
state.cash = 1000.0
state.cb_cash = 0.0
state.pnl = {}
state.history = []
state.tracebacks = []
state.last_updated = time.time()
state.lock = threading.Lock()
state.trading_enabled = True
state.halted = False
state.base_equity = None
state.wallets = ROTATE_WALLETS[:]
state.wallet_index = 0
state.current_wallet = state.wallets[state.wallet_index] if state.wallets else WALLET_ADDRESS
LOG_RING = deque(maxlen=500)
_price_cache = {}

# Fusion state
state.last_fusion_meta = {}
state.last_grok = None
state.fusion_installed_at = time.time()
state.last_reweight_at = None
state.fusion_weights = {"ghost": 0.30, "claude": 0.30, "gemini": 0.20, "grok": 0.20}
state.fusion_stats = {
    "ghost":  {"calls":0,"wins":0,"losses":0,"roi_sum":0.0},
    "claude": {"calls":0,"wins":0,"losses":0,"roi_sum":0.0},
    "gemini": {"calls":0,"wins":0,"losses":0,"roi_sum":0.0},
    "grok":   {"calls":0,"wins":0,"losses":0,"roi_sum":0.0},
}

# ============================== LOGGING ==============================

def _log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    line = f"{ts} | {msg}"
    LOG_RING.append(line)
    logging.info(line)

# ============================== AUTH =================================

def _authed_query():
    tok = request.args.get("token")
    return tok == TOKEN

def require_auth(fn):
    @wraps(fn)
    def wrap(*a, **k):
        hdr = request.headers.get("Authorization")
        if hdr == TOKEN:
            return fn(*a, **k)
        # allow GET fallback via query token for convenience
        if request.method == 'GET' and _authed_query():
            return fn(*a, **k)
        return jsonify({"error":"unauthorized"}), 403
    return wrap

# ============================== DATA SRCS ============================

def price_primary(sym):
    gid = COINGECKO_IDS.get(sym)
    if not gid: return -1.0
    r = requests.get("https://api.coingecko.com/api/v3/simple/price", params={"ids":gid, "vs_currencies":"usd"}, timeout=6)
    r.raise_for_status()
    v = r.json().get(gid,{}).get("usd")
    return float(v) if v is not None else -1.0

def price_fallback(sym):
    pid = COINPAPRIKA_IDS.get(sym)
    if not pid: return -1.0
    r = requests.get(f"https://api.coinpaprika.com/v1/tickers/{pid}", timeout=6)
    r.raise_for_status()
    v = r.json().get('quotes',{}).get('USD',{}).get('price')
    return float(v) if v is not None else -1.0

def get_price(sym):
    now = time.time()
    c = _price_cache.get(sym)
    if c and now - c['ts'] < PRICE_CACHE_SEC:
        return c['price']
    val = -1.0
    try:
        val = price_primary(sym)
    except Exception as e:
        _log(f"price primary err {sym}: {e}")
    if val <= 0:
        try:
            val = price_fallback(sym)
        except Exception as e:
            _log(f"price fallback err {sym}: {e}")
    if val > 0:
        _price_cache[sym] = {"price": val, "ts": now}
    return val

def get_eth_balance_usd(wallet):
    try:
        url = (f"https://api.covalenthq.com/v1/1/address/{wallet}/balances_v2/"
               f"?quote-currency=USD&nft=false&no-nft-fetch=true&key={COVALENT_KEY}")
        res = requests.get(url, timeout=8)
        js = res.json(); items = js.get("data",{}).get("items",[])
        for it in items:
            if it.get("contract_ticker_symbol") == "ETH":
                q = it.get("quote") or 0
                return round(float(q), 2)
    except Exception as e:
        _log(f"Covalent err: {e}")
    return 0.0

def get_coinbase_balance_and_positions():
    try:
        headers = {"CB-ACCESS-KEY": COINBASE_API_KEY, "CB-ACCESS-SIGN": "", "CB-ACCESS-TIMESTAMP": str(int(time.time())), "CB-VERSION": "2023-01-01"}
        r = requests.get("https://api.coinbase.com/v2/accounts", headers=headers, timeout=8)
        data = r.json().get("data", []) if r.status_code == 200 else []
        usd_total, positions = 0.0, []
        for a in data:
            cur = a.get('balance',{}).get('currency'); amt = float(a.get('balance',{}).get('amount',0))
            if not cur or amt == 0: continue
            if cur == 'USD': usd_total += amt
            else: positions.append({"symbol": cur, "amount": amt})
        return round(usd_total,2), positions
    except Exception as e:
        _log(f"Coinbase err: {e}")
        return 0.0, []

# ============================== MEMORY ===============================

def ghostmirror_update(sym, score):
    mem = state.diag["memory"].setdefault(sym, deque(maxlen=40))
    try:
        score = round(float(score), 3)
    except Exception:
        score = 0.0
    mem.append(score)

# ======================== SHEETS/CSV LOGGING =========================

def _ensure_csv_header(path):
    try:
        if not os.path.exists(path):
            with open(path, "w") as f:
                f.write("Timestamp,ISO,Coin,Action,Price,Amount,GhostScore,ClaudeScore,GeminiScore,GrokScore,FusionPass\n")
    except Exception as e:
        _log(f"CSV header error: {e}")

def _append_csv_row(path, row):
    try:
        _ensure_csv_header(path)
        with open(path, "a") as f:
            f.write(",".join([str(x) if x is not None else "" for x in row]) + "\n")
    except Exception as e:
        _log(f"CSV append error: {e}")

def _init_sheets():
    if not gspread:
        _log("Sheets: gspread not installed; using CSV fallback")
        return
    if not GOOGLE_SHEETS_ID or not GOOGLE_SERVICE_ACCOUNT_JSON_B64:
        _log("Sheets: GOOGLE_SHEETS_ID/GOOGLE_SERVICE_ACCOUNT_JSON_B64 not set; CSV fallback")
        return
    try:
        creds = json.loads(base64.b64decode(GOOGLE_SERVICE_ACCOUNT_JSON_B64))
        client = gspread.service_account_from_dict(creds)
        sh = client.open_by_key(GOOGLE_SHEETS_ID)
        try:
            ws = sh.worksheet("Trades")
        except Exception:
            ws = sh.add_worksheet(title="Trades", rows="1", cols="11")
            ws.append_row(["Timestamp","ISO","Coin","Action","Price","Amount","GhostScore","ClaudeScore","GeminiScore","GrokScore","FusionPass"], value_input_option="USER_ENTERED")
        state.gs = {"client": client, "sheet": ws, "ready": True}
        _log("Sheets: connected to Trades worksheet")
    except Exception as e:
        _log(f"Sheets init error: {e}")


def _fmt_ts(ts):
    if not ts:
        return None
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts))
    except Exception:
        return None


def log_trade_row(sym, action, price, amount, fmeta=None):
    ts = int(time.time())
    iso = _fmt_ts(ts)
    g = c = m = k = ""
    fusion_pass = ""
    if fmeta:
        sc = fmeta.get("scores", {})
        gv = sc.get("ghost"); cv = sc.get("claude"); mv = sc.get("gemini"); kv = sc.get("grok")
        g = round(gv, 4) if isinstance(gv, (int, float)) else (gv if gv is not None else "")
        c = round(cv, 4) if isinstance(cv, (int, float)) else (cv if cv is not None else "")
        m = round(mv, 4) if isinstance(mv, (int, float)) else (mv if mv is not None else "")
        k = round(kv, 4) if isinstance(kv, (int, float)) else (kv if kv is not None else "")
        fusion_pass = "YES" if fmeta.get("approved") else "NO"
    row = [ts, iso, sym, action, price, amount, g, c, m, k, fusion_pass]

    if getattr(state, "gs", {}).get("ready"):
        try:
            state.gs["sheet"].append_row(row, value_input_option="USER_ENTERED")
        except Exception as e:
            _log(f"Sheets append error: {e}; writing CSV fallback")
            _append_csv_row("trades.csv", row)
    else:
        _append_csv_row("trades.csv", row)

# ============================== GROK (xAI) SENTIMENT ==================

def _grok_hype_score(sym, context=None):
    if not (GROK_ENABLE and GROK_API_KEY):
        return None
    try:
        prompt = (
            "You are scoring crypto meme coin hype. "
            "Return ONLY a JSON object with key 'hype' between 0 and 1. "
            f"Symbol: {sym}. Consider recent social buzz, novelty, and virality likelihood."
        )
        headers = {"Authorization": f"Bearer {GROK_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": GROK_MODEL,
            "messages": [
                {"role": "system", "content": "Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        r = requests.post(GROK_ENDPOINT, headers=headers, json=payload, timeout=GROK_TIMEOUT)
        if r.status_code != 200:
            _log(f"Grok API HTTP {r.status_code}")
            return None
        js = r.json()
        try:
            content = js["choices"][0]["message"]["content"].strip()
        except Exception:
            return None
        try:
            if content.startswith("{") and content.endswith("}"):
                data = json.loads(content)
            else:
                lb = content.find("{"); rb = content.rfind("}")
                data = json.loads(content[lb:rb+1]) if lb != -1 and rb != -1 else {}
        except Exception:
            return None
        hype = float(data.get("hype"))
        return max(0.0, min(1.0, hype))
    except Exception as e:
        _log(f"Grok error: {e}")
        return None

# ============================== FUSION CORE ===========================

# --- Fusion AI debate (stub/sim) ---

def _simulate_ai_scores(sym, base_score):
    # Map base score (0.4..2.0) to 0..1 confidence and add small noise per AI
    x = max(0.0, min(1.0, (base_score - 0.4) / (2.0 - 0.4)))
    rn = random.random()
    ghost = max(0.0, min(1.0, x + (0.10*rn - 0.05)))
    claude = max(0.0, min(1.0, x + (0.12*random.random() - 0.06)))
    gemini = max(0.0, min(1.0, x + (0.12*random.random() - 0.06)))
    return {"ghost": ghost, "claude": claude, "gemini": gemini}


def fusion_consensus(sym, base_signal_score):
    """Returns (approved:boolean, scores:dict, weighted_sum:float)."""
    if not FUSION_MODE:
        return True, {"ghost":1.0, "claude":1.0, "gemini":1.0, "grok":1.0}, 1.0

    # Opinions (stub unless real APIs wired)
    scores = _simulate_ai_scores(sym, base_signal_score) if FUSION_SIM else _simulate_ai_scores(sym, base_signal_score)

    # Add Grok as fourth voter if available
    gscore = _grok_hype_score(sym)
    if gscore is not None:
        state.last_grok = gscore
        scores["grok"] = gscore

    # Weighted average over whichever voters are present
    total_w = 0.0; total = 0.0; used = []
    for k, w in state.fusion_weights.items():
        s = scores.get(k)
        if s is None:
            continue
        used.append(k)
        total_w += float(w)
        total += float(s) * float(w)
    weighted = (total / total_w) if total_w > 0 else 0.0

    approved = (weighted >= FUSION_APPROVAL_THRESHOLD)

    # Count a call for each contributing AI
    for k in used:
        state.fusion_stats[k]["calls"] += 1

    return approved, scores, weighted


def recompute_fusion_weights():
    """Re-weight AI based on simple performance heuristic (wins/losses + ROI)."""
    try:
        stats = state.fusion_stats
        # Score each AI = win_rate * 0.7 + avg_roi_norm * 0.3
        scores = {}
        for k, s in stats.items():
            calls = max(1, (s.get("wins",0) + s.get("losses",0)))
            win_rate = s.get("wins",0) / calls
            avg_roi = (s.get("roi_sum",0.0) / calls)
            # normalize avg_roi into 0..1 softly
            roi_norm = max(0.0, min(1.0, 0.5 + avg_roi))
            scores[k] = 0.7*win_rate + 0.3*roi_norm
        # prevent all-zero
        total = sum(scores.values()) or 1.0
        new_w = {k: max(0.05, v/total) for k, v in scores.items()}
        # re-normalize
        ssum = sum(new_w.values())
        new_w = {k: v/ssum for k, v in new_w.items()}
        state.fusion_weights = new_w
        state.last_reweight_at = time.time()
        _log(f"Fusion re-weight -> {state.fusion_weights}")
    except Exception as e:
        _log(f"recompute_fusion_weights error: {e}")


def _fusion_reweight_due():
    now = time.time()
    if state.last_reweight_at is None:
        return (now - state.fusion_installed_at) >= FUSION_REWEIGHT_HOURS*3600
    return (now - state.last_reweight_at) >= FUSION_REWEIGHT_HOURS*3600


def _fusion_min_trades_met():
    total_calls = sum(s["calls"] for s in state.fusion_stats.values())
    if state.last_reweight_at is None:
        return total_calls >= FUSION_MIN_TRADES_FOR_REWEIGHT
    return total_calls >= FUSION_MIN_TRADES_FOR_REWEIGHT


def fusion_reweight_loop():
    while True:
        try:
            if _fusion_reweight_due() and _fusion_min_trades_met():
                recompute_fusion_weights()
        except Exception as e:
            _log(f"Fusion re-weight error: {e}")
        time.sleep(60)

# ============================== TRADING ===============================

def dynamic_alloc_frac():
    try:
        if state.base_equity is None: return ALLOC_BASE_FRAC
        last_eq = state.diag["equity"][-1] if state.diag["equity"] else state.base_equity
        growth = max(0.0, (last_eq - state.base_equity) / max(1.0, state.base_equity))
        bump = min(0.10, 0.5*growth)
        return max(0.05, min(0.50, ALLOC_BASE_FRAC + bump))
    except Exception:
        return ALLOC_BASE_FRAC

def buy(sym, score, fusion_meta=None):
    if not state.trading_enabled or state.halted: return
    px = get_price(sym)
    if px <= 0: return
    with state.lock:
        alloc = dynamic_alloc_frac(); budget = state.cash * alloc
        if budget < 1: return
        qty = round(budget/px, 6)
        state.positions[sym] = Pos(sym, qty, px)
        state.cash -= qty*px
        state.pnl[sym] = state.pnl.get(sym, 0.0) - qty*px
        state.history.append(("BUY", sym, qty, px))
        extra = ""
        if fusion_meta:
            sc = fusion_meta.get('scores', {})
            g_s = sc.get('ghost', 0.0); c_s = sc.get('claude', 0.0); m_s = sc.get('gemini', 0.0); k_s = sc.get('grok')
            extra = f" | Fusion {fusion_meta['weighted']:.2f} (G:{g_s:.2f}, C:{c_s:.2f}, M:{m_s:.2f}" + (f", K:{k_s:.2f}" if isinstance(k_s,(int,float)) else "") + ")"
            state.last_fusion_meta[sym] = fusion_meta
        state.msg = f"BUY {sym} @ ${px:.6f} ({qty}u) — alloc {alloc:.2f}{extra}"
        _log(state.msg)
        try:
            log_trade_row(sym, "BUY", px, qty, fusion_meta)
        except Exception as e:
            _log(f"log_trade_row BUY error: {e}")

def sell(sym, reason="EXIT"):
    if sym not in state.positions: return
    px = get_price(sym)
    if px <= 0: return
    with state.lock:
        p = state.positions.pop(sym)
        proceeds = p.qty*px
        state.cash += proceeds
        pnl_delta = proceeds - (p.qty*p.price)
        state.pnl[sym] = state.pnl.get(sym, 0.0) + proceeds
        state.history.append(("SELL", sym, p.qty, px))
        state.msg = f"{reason} {sym} @ ${px:.6f} ({p.qty}u)"
        _log(state.msg)
        # Update Fusion stats win/loss by realized PnL sign
        outcome = "wins" if pnl_delta > 0 else "losses"
        for k in state.fusion_stats:
            state.fusion_stats[k][outcome] += 1
            try:
                roi = (px - p.price) / max(1e-9, p.price)
            except Exception:
                roi = 0.0
            state.fusion_stats[k]["roi_sum"] += roi
        try:
            fmeta = state.last_fusion_meta.get(sym)
            log_trade_row(sym, "SELL", px, p.qty, fmeta)
        except Exception as e:
            _log(f"log_trade_row SELL error: {e}")
        finally:
            state.last_fusion_meta.pop(sym, None)


def risk_checks_and_trailing():
    for sym, p in list(state.positions.items()):
        cur = get_price(sym)
        if cur <= 0: continue
        if cur > p.peak: p.peak = cur
        if cur >= p.price*(1+TAKE_PROFIT_PCT):
            sell(sym, reason="TP"); continue
        if cur <= p.price*(1-STOP_LOSS_PCT):
            sell(sym, reason="SL"); continue

# ============================== UPDATE ENGINE ========================

def _sha256(data: bytes):
    h = hashlib.sha256(); h.update(data); return h.hexdigest()


def maybe_check_update():
    if not UPDATE_URL or BACKGROUND_UPDATE_HOURS <= 0:
        return
    try:
        headers = {"User-Agent":"GhostFusion/2.6"}
        if UPDATE_BEARER:
            headers["Authorization"] = f"Bearer {UPDATE_BEARER}"
        r = requests.get(UPDATE_URL, timeout=10, headers=headers)
        if r.status_code != 200:
            _log(f"Update check: HTTP {r.status_code}")
            return
        payload = r.json() if r.headers.get('content-type','').startswith('application/json') else None
        if not payload:
            _log("Update check: expected JSON payload {version, code or code_b64}")
            return
        new_ver = payload.get("version")
        code_b64 = payload.get("code_b64")
        code_txt = payload.get("code")
        blob = base64.b64decode(code_b64) if code_b64 else (code_txt.encode('utf-8') if code_txt else None)
        if not blob:
            _log("Update check: no code provided")
            return
        sha_ok = True
        if payload.get("sha256"):
            sha_ok = (payload.get("sha256").lower() == _sha256(blob).lower())
        if not sha_ok:
            _log("Update check: sha256 mismatch; aborting")
            return
        # Compare with existing file hash
        try:
            with open(sys.argv[0], 'rb') as f:
                cur = f.read()
            if _sha256(cur) == _sha256(blob):
                _log("Update check: already up to date")
                return
        except Exception:
            pass
        # Write new file
        with open("main.py.new", 'wb') as f:
            f.write(blob)
        _log(f"Update downloaded: {new_ver} -> main.py.new")
        if AUTO_APPLY_UPDATE:
            try:
                os.replace("main.py.new", sys.argv[0])
                _log("Update applied. Restarting...")
                os.execv(sys.executable, [sys.executable] + sys.argv)
            except Exception as e:
                _log(f"Auto-apply failed: {e}")
                return
    except Exception as e:
        _log(f"Update check error: {e}")


def updater_loop():
    if not UPDATE_URL or BACKGROUND_UPDATE_HOURS <= 0:
        return
    while True:
        try:
            maybe_check_update()
        except Exception as e:
            _log(f"Updater error: {e}")
        time.sleep(max(60, BACKGROUND_UPDATE_HOURS*3600))

# ============================== ROUTES ================================
@app.get("/")
def home():
    return redirect(f"/view8?token={TOKEN}")

@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": int(time.time()), "version": VERSION})

@app.get("/api/logs")
@require_auth
def api_logs():
    return jsonify({"ok": True, "n": len(LOG_RING), "items": list(LOG_RING)})

@app.get("/api/version")
@require_auth
def api_version():
    return jsonify({"version": VERSION, "fusion_mode": FUSION_MODE})

@app.post("/api/update_now")
@require_auth
def api_update_now():
    maybe_check_update()
    return jsonify({"ok": True})

@app.get("/view8")
@require_auth
def view8():
    state.cb_cash, state.diag["cb_holdings"] = get_coinbase_balance_and_positions()
    state.cash = get_eth_balance_usd(state.current_wallet)
    total_cash = state.cash + state.cb_cash

    held = {s: {"qty": p.qty, "price": p.price} for s,p in state.positions.items()}
    prices = {s: get_price(s) for s in TRACKED}
    equity = total_cash + sum(h["qty"]*max(0.0, prices.get(s,0.0)) for s,h in held.items())
    if state.base_equity is None: state.base_equity = equity
    state.diag["equity"].append(equity)

    equity_list = list(state.diag["equity"])
    labels_list = list(range(1, len(equity_list)+1))
    signals = "<br>".join([f"{t} {s} @ ${p:.6f} ({q}u)" for t,s,q,p in state.history[-8:]]) or "—"
    held_box = "<br>".join([f"{s}: {d['qty']} @ ${d['price']:.6f}" for s,d in held.items()]) or "None"
    cb_box = "<br>".join([f"{x['symbol']}: {x['amount']}" for x in state.diag.get('cb_holdings',[])]) or "None"

    trade_badge = "ON" if state.trading_enabled else "OFF"
    halted_badge = "PAUSED (low cash)" if state.halted else "RUNNING"

    # Fusion panel derived values
    weights = state.fusion_weights
    last_rw = _fmt_ts(state.last_reweight_at)
    next_rw = None
    base_ts = state.last_reweight_at if state.last_reweight_at else state.fusion_installed_at
    next_ts = base_ts + FUSION_REWEIGHT_HOURS*3600 if FUSION_REWEIGHT_HOURS else None
    if next_ts:
        next_rw = _fmt_ts(next_ts)

    html = f"""
    <html><head><title>Ghost Bridge {VERSION}</title><meta name='viewport' content='width=device-width'>
    <script src='https://cdn.jsdelivr.net/npm/chart.js'></script>
    <style>
      body{{margin:0;background:#003b00;color:#fff;font-family:ui-monospace,Menlo,Consolas,monospace}}
      .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:18px;padding:20px}}
      .card{{border:1px solid #0f0;background:#002700;border-radius:12px;padding:14px}}
      .btn{{display:inline-block;margin:4px 6px;padding:8px 10px;border:1px solid #0f0;border-radius:10px;color:#0f0;background:#001a00;cursor:pointer}}
      .btn:hover{{background:#013501}}
      .kv small{{opacity:.8}}
      .pill{{display:inline-block;padding:2px 8px;border-radius:999px;border:1px solid #0f0;margin-left:6px}}
    </style>
    </head><body>
    <div class='grid'>
      <div class='card'>
        <div class='kv'><small>Total Equity</small></div>
        <div style='font-size:30px;' id='eq_val'>${equity:,.2f}</div>
        <div style='background:#133;height:10px;border-radius:8px;margin-top:8px'>
          <div id='eq_bar' style='background:#0f0;height:10px;border-radius:8px;width:{min(100.0, (equity/100)):.2f}%'></div>
        </div>
        <div style='margin-top:8px'>Target $10,000</div>
      </div>

      <div class='card'>
        <div class='kv'><small>Engine</small></div>
        <div>Feed: sim</div>
        <div>Status: {halted_badge}</div>
        <div>Trading: <b id='trade_badge'>{trade_badge}</b></div>
        <div style='margin-top:8px'>
          <button class='btn' onclick='toggleTrading()'>Toggle Trading</button>
          <button class='btn' onclick='rotateWallet()'>Rotate Wallet</button>
          <button class='btn' onclick='quickTP()'>TP+{int(TAKE_PROFIT_PCT*100)}%</button>
          <button class='btn' onclick='quickSL()'>SL-{int(STOP_LOSS_PCT*100)}%</button>
        </div>
      </div>

      <div class='card'>
        <div class='kv'><small>Wallet (MetaMask)</small></div>
        <div id='mm_addr'>{state.current_wallet}</div>
        <div>USD: $<span id='mm_usd'>{state.cash:,.2f}</span></div>
      </div>

      <div class='card'>
        <div class='kv'><small>Wallet (Coinbase)</small></div>
        <div>USD: $<span id='cb_usd'>{state.cb_cash:,.2f}</span></div>
        <div style='margin-top:6px'><small>Holdings:</small><br><span id='cb_box'>{cb_box}</span></div>
      </div>

      <div class='card'>
        <div class='kv'><small>Positions</small></div>
        <div id='pos_box'>{held_box}</div>
      </div>

      <div class='card'>
        <div class='kv'><small>Signal Feed</small></div>
        <div style='max-height:160px;overflow:auto' id='sig_box'>{signals}</div>
      </div>

      <div class='card'>
        <div class='kv'><small>Fusion AI Influence</small><span class='pill' id='fusion_mode_pill'>{'ON' if FUSION_MODE else 'OFF'}</span></div>
        <div id='fusion_weights'>Ghost {int(weights['ghost']*100)}% · Claude {int(weights['claude']*100)}% · Gemini {int(weights['gemini']*100)}%</div>
        <div style='margin-top:6px'>Last: <span id='fusion_last'>{last_rw or '—'}</span> | Next: <span id='fusion_next'>{next_rw or '—'}</span></div>
        <div style='margin-top:8px'>
          <button class='btn' onclick='toggleFusion()'>Toggle Fusion</button>
          <button class='btn' onclick='reweightNow()'>Re-weight Now</button>
          <button class='btn' onclick='updateNow()'>Update Now</button>
        </div>
      </div>

      <div class='card' style='grid-column:1/-1'>
        <canvas id='eq' height='110'></canvas>
      </div>
    </div>
    <script>
    async function post(url, body={{}}){{ body.token='{TOKEN}'; const r = await fetch(url+'?token={TOKEN}', {{method:'POST', headers:{{'Content-Type':'application/json','Authorization':'{TOKEN}'}}, body:JSON.stringify(body)}}); return r.json(); }}
    async function get(url){{ const r = await fetch(url+'?token={TOKEN}', {{headers:{{'Authorization':'{TOKEN}'}}}}); return r.json(); }}
    async function toggleTrading(){{ const d = await post('/api/toggle_trading'); document.getElementById('trade_badge').innerText = d.enabled?'ON':'OFF'; }}
    async function rotateWallet(){{ const d = await post('/api/rotate_wallet'); document.getElementById('mm_addr').innerText = d.wallet; }}
    async function quickTP(){{ const d = await post('/api/config', {{take_profit: {TAKE_PROFIT_PCT}}}); alert('TP: '+Math.round(d.take_profit*100)+'%'); }}
    async function quickSL(){{ const d = await post('/api/config', {{stop_loss: {STOP_LOSS_PCT}}}); alert('SL: '+Math.round(d.stop_loss*100)+'%'); }}

    async function toggleFusion(){{ const d = await post('/api/toggle_fusion'); document.getElementById('fusion_mode_pill').innerText = d.fusion_mode?'ON':'OFF'; }}
    async function reweightNow(){{ const d = await post('/api/reweight_now'); alert('Re-weighted: '+(d.ts || 'now')); pullFusion(); }}
    async function updateNow(){{ const d = await post('/api/update_now'); alert('Update check triggered'); }}

    async function pull(){{
      try{{
        const d = await get('/api/status');
        document.getElementById('eq_val').innerText = '$'+Number(d.equity||0).toLocaleString();
        document.getElementById('eq_bar').style.width = Math.min(100,(d.equity||0)/100).toFixed(2)+'%';
        document.getElementById('mm_usd').innerText = Number(d.cash_mm||0).toLocaleString();
        document.getElementById('cb_usd').innerText = Number(d.cash_cb||0).toLocaleString();
      }}catch(e){{}}
    }}

    async function pullFusion(){{
      try{{
        const f = await get('/api/fusion_status');
        const fw = f.weights || {{ghost:0.35,claude:0.40,gemini:0.25}};
        document.getElementById('fusion_mode_pill').innerText = f.fusion_mode?'ON':'OFF';
        document.getElementById('fusion_weights').innerText = 'Ghost '+Math.round(fw.ghost*100)+'% · Claude '+Math.round(fw.claude*100)+'% · Gemini '+Math.round(fw.gemini*100)+'%';
        document.getElementById('fusion_last').innerText = f.last_reweight_at_str || '—';
        document.getElementById('fusion_next').innerText = f.next_reweight_at_str || '—';
      }}catch(e){{}}
    }}
    setInterval(pull, 4000);
    setInterval(pullFusion, 4000);

    const ctx = document.getElementById('eq').getContext('2d');
    const labels = {labels_list};
    const data = {equity_list};
    new Chart(ctx, {{type:'line', data:{{labels:labels, datasets:[{{label:'Equity', data:data, borderColor:'lime', tension:0.35}}]}}, options:{{responsive:true, scales:{{y:{{beginAtZero:true}}}}}}}});
    </script>
    </body></html>
    """
    return html

@app.get("/api/status")
@require_auth
def api_status():
    last_eq = state.diag["equity"][-1] if state.diag["equity"] else 0
    return jsonify({
        "version": VERSION,
        "msg": state.msg,
        "trading_enabled": state.trading_enabled,
        "halted": state.halted,
        "wallet": state.current_wallet,
        "cash_mm": state.cash,
        "cash_cb": state.cb_cash,
        "equity": last_eq,
        "held": {s:{"qty":p.qty,"price":p.price} for s,p in state.positions.items()},
        "cb_holdings": state.diag.get("cb_holdings", []),
        "tracebacks": state.tracebacks[-5:],
    })

@app.get("/api/fusion_status")
@require_auth
def api_fusion_status():
    weights = getattr(state, "fusion_weights", {"ghost": 0.35, "claude": 0.40, "gemini": 0.25, "grok": 0.0})
    stats = getattr(state, "fusion_stats", {
        "ghost":  {"calls": 0, "wins": 0, "losses": 0, "roi_sum": 0.0},
        "claude": {"calls": 0, "wins": 0, "losses": 0, "roi_sum": 0.0},
        "gemini": {"calls": 0, "wins": 0, "losses": 0, "roi_sum": 0.0},
        "grok":   {"calls": 0, "wins": 0, "losses": 0, "roi_sum": 0.0},
    })
    installed_at = getattr(state, "fusion_installed_at", time.time())
    last_reweight_at = getattr(state, "last_reweight_at", None)

    reweight_hours = globals().get("FUSION_REWEIGHT_HOURS", 24)
    min_trades = globals().get("FUSION_MIN_TRADES_FOR_REWEIGHT", 5)
    fusion_mode = bool(globals().get("FUSION_MODE", True))

    ai_calls_total = sum(int(s.get("calls", 0)) for s in stats.values())
    base_ts = last_reweight_at if last_reweight_at else installed_at
    next_reweight_at = base_ts + reweight_hours * 3600 if reweight_hours else None

    last_eq = state.diag["equity"][-1] if state.diag.get("equity") else 0.0
    open_positions = {s: {"qty": p.qty, "price": p.price} for s, p in state.positions.items()}

    return jsonify({
        "ok": True,
        "version": VERSION,
        "fusion_mode": fusion_mode,
        "weights": weights,
        "stats": stats,
        "ai_calls_total": ai_calls_total,
        "reweight_interval_hours": reweight_hours,
        "min_trades_for_reweight": min_trades,
        "installed_at": int(installed_at),
        "installed_at_str": _fmt_ts(installed_at),
        "last_reweight_at": (int(last_reweight_at) if last_reweight_at else None),
        "last_reweight_at_str": _fmt_ts(last_reweight_at),
        "next_reweight_at": (int(next_reweight_at) if next_reweight_at else None),
        "next_reweight_at_str": _fmt_ts(next_reweight_at),
        "last_grok": state.last_grok,
        "engine": {
            "trading_enabled": state.trading_enabled,
            "halted": state.halted,
            "wallet": getattr(state, "current_wallet", None),
        },
        "portfolio": {
            "equity": last_eq,
            "positions": open_positions,
            "history_n": len(state.history),
        }
    })

@app.post("/api/toggle_trading")
@require_auth
def api_toggle_trading():
    state.trading_enabled = not state.trading_enabled
    return jsonify({"enabled": state.trading_enabled})

@app.post("/api/rotate_wallet")
@require_auth
def api_rotate_wallet():
    if not state.wallets:
        return jsonify({"ok": False, "reason": "no wallets configured"}), 400
    state.wallet_index = (state.wallet_index + 1) % len(state.wallets)
    state.current_wallet = state.wallets[state.wallet_index]
    return jsonify({"ok": True, "wallet": state.current_wallet, "index": state.wallet_index})

@app.get("/api/config")
@require_auth
def api_get_cfg():
    return jsonify({
        "alloc_base_frac": ALLOC_BASE_FRAC,
        "take_profit": TAKE_PROFIT_PCT,
        "stop_loss": STOP_LOSS_PCT,
        "low_cash_usd_threshold": LOW_CASH_USD_THRESHOLD,
        "price_cache_sec": PRICE_CACHE_SEC,
        "fusion_mode": FUSION_MODE,
        "fusion_reweight_hours": FUSION_REWEIGHT_HOURS,
    })

@app.post("/api/config")
@require_auth
def api_set_cfg():
    global ALLOC_BASE_FRAC, TAKE_PROFIT_PCT, STOP_LOSS_PCT, PRICE_CACHE_SEC, FUSION_REWEIGHT_HOURS
    d = request.get_json(force=True, silent=True) or {}
    if "alloc_base_frac" in d:
        try: ALLOC_BASE_FRAC = float(d["alloc_base_frac"])
        except Exception: pass
    if "take_profit" in d:
        try: TAKE_PROFIT_PCT = float(d["take_profit"])
        except Exception: pass
    if "stop_loss" in d:
        try: STOP_LOSS_PCT = float(d["stop_loss"]) 
        except Exception: pass
    if "price_cache_sec" in d:
        try: PRICE_CACHE_SEC = int(d["price_cache_sec"]) 
        except Exception: pass
    if "fusion_reweight_hours" in d:
        try: FUSION_REWEIGHT_HOURS = int(d["fusion_reweight_hours"]) 
        except Exception: pass
    return api_get_cfg()

@app.post("/api/toggle_fusion")
@require_auth
def api_toggle_fusion():
    global FUSION_MODE
    FUSION_MODE = not FUSION_MODE
    state.msg = f"Ghost Protocol {VERSION} — Fusion ON" if FUSION_MODE else f"Ghost Protocol {VERSION} — Fusion OFF"
    return jsonify({"ok": True, "fusion_mode": FUSION_MODE})

@app.post("/api/reweight_now")
@require_auth
def api_reweight_now():
    recompute_fusion_weights()
    return jsonify({"ok": True, "weights": state.fusion_weights, "ts": int(state.last_reweight_at)})

# ============================== LOOP =================================

def core_step():
    state.halted = (state.cash < LOW_CASH_USD_THRESHOLD)
    sym = random.choice(TRACKED)
    score = round(random.uniform(0.4, 2.0), 2)
    ghostmirror_update(sym, score)

    if state.trading_enabled and not state.halted:
        approved = True; fmeta = None
        if FUSION_MODE:
            ok, scores, weighted = fusion_consensus(sym, score)
            approved = ok
            fmeta = {"scores": scores, "weighted": weighted, "approved": ok}
        if approved and (score > 1.5) and (sym not in state.positions):
            buy(sym, score, fusion_meta=fmeta)
        elif (score < 0.6) and (sym in state.positions):
            sell(sym, reason="EXIT")

    risk_checks_and_trailing()


def safe_loop():
    while True:
        try:
            core_step()
        except Exception:
            state.tracebacks.append(traceback.format_exc())
            if len(state.tracebacks) > 10: state.tracebacks.pop(0)
        time.sleep(8)

# ============================== ENTRY ================================
if __name__ == '__main__':
    try:
        _init_sheets()
    except Exception as e:
        _log(f"Sheets init at startup failed: {e}")
    threading.Thread(target=safe_loop, daemon=True).start()
    threading.Thread(target=fusion_reweight_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=8080)


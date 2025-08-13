# main.py — Ghost Mini (single‑file, copy‑paste friendly)
# - Runs out-of-the-box (SIM) with a heartbeat + UI
# - Optional Coinbase Advanced Trade overlay if creds are present
# - Simple advisor loop, safe defaults, and /health + /api/status
# - No secrets required to start

import os, time, math, random, json, logging, threading, hmac, hashlib, base64
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from datetime import datetime, timezone, date
import requests
from flask import Flask, request, jsonify, Response, redirect

# =============================== CONFIG ===============================
VERSION = "mini-1.0.0"

def _env_bool(k, d=False):
    v = os.getenv(k, str(d))
    return str(v).lower() in ("1", "true", "yes", "y")

# Advisor & runtime
ADVISOR_MODE     = os.getenv("ADVISOR_MODE", "opportunistic")   # opportunistic | conservative
PAPER_TRADE      = _env_bool("PAPER_TRADE", True)               # True boots even w/o creds
EXECUTION_BACKEND= os.getenv("EXECUTION_BACKEND", "sim")        # sim | coinbase
CB_MODE          = os.getenv("CB_MODE", "read")                 # read | trade
CB_SANDBOX       = _env_bool("CB_SANDBOX", False)

LOOP_SECONDS     = int(os.getenv("LOOP_SECONDS", "5"))
PORT             = int(os.getenv("PORT", "3000"))
TOKEN            = os.getenv("TOKEN", "changeme")
POLL_SEC         = int(os.getenv("POLL_SEC", "2"))

# Goal panel (cosmetic)
START_EQUITY     = float(os.getenv("START_EQUITY", "1000"))
GOAL_EQUITY      = float(os.getenv("GOAL_EQUITY", "10000"))
GOAL_DEADLINE    = os.getenv("GOAL_DEADLINE_ISO", "2025-12-31")

# Universe
STABLE           = os.getenv("STABLE", "USDC").upper()
UNIVERSE         = [s.strip().upper() for s in os.getenv("UNIVERSE", "BTC,ETH,PEPE,DOGE,SHIB").split(",") if s.strip()]
if STABLE not in UNIVERSE: UNIVERSE.append(STABLE)

TOP_N            = int(os.getenv("TOP_N", "3"))
MIN_SCORE_ENTER  = float(os.getenv("MIN_SCORE_ENTER", "1.00"))
MIN_SCORE_EXIT   = float(os.getenv("MIN_SCORE_EXIT", "0.85"))
REBAL_EVERY_TICKS= int(os.getenv("REBAL_EVERY_TICKS", "12"))
WARMUP_TICKS     = int(os.getenv("WARMUP_TICKS", "60"))

# Risk (simple)
MAX_GROSS_EXPOSURE = float(os.getenv("MAX_GROSS_EXPOSURE", "0.80"))
MAX_PER_TRADE_FRAC = float(os.getenv("MAX_PER_TRADE_FRAC", "0.35"))
MIN_TRADE_USD      = float(os.getenv("MIN_TRADE_USD", "50"))

# Coinbase creds (optional)
CB_API_KEY     = os.getenv("CB_API_KEY", "").strip()
CB_API_SECRET  = os.getenv("CB_API_SECRET", "").strip()   # entire PEM string OK
COINBASE_PRODUCTS = [p.strip() for p in os.getenv("COINBASE_PRODUCTS", "BTC-USD,ETH-USD,PEPE-USD,DOGE-USD,SHIB-USD,USDC-USD").split(",") if p.strip()]

# =============================== Coinbase client ===============================
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class CoinbaseClient:
    """
    Advanced Trade client with retry/backoff and safe base‑URL handling.
    Works in production by default; set CB_SANDBOX=true to switch.
    """

    def __init__(self, key, secret, sandbox=False, base_url=None, timeout=12):
        self.key = (key or "").strip()
        self.secret = (secret or "").strip()

        env_url = (os.getenv("COINBASE_API_URL") or "").strip()
        if not base_url:
            base_url = env_url or ("https://api.sandbox.coinbase.com" if sandbox else "https://api.coinbase.com")
        if "sandbox" in base_url.lower() and not sandbox:
            base_url = "https://api.coinbase.com"

        self.base_v3 = base_url.rstrip("/") + "/api/v3/brokerage"
        self.timeout = timeout

        self.s = requests.Session()
        retry = Retry(
            total=6, connect=3, read=3, backoff_factor=0.6,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET", "POST"]),
        )
        self.s.mount("https://", HTTPAdapter(max_retries=retry))
        self.s.mount("http://",  HTTPAdapter(max_retries=retry))

    def _sig(self, ts: str, method: str, path: str, body: str = "") -> str:
        msg = f"{ts}{method.upper()}{path}{body}".encode("utf-8")
        mac = hmac.new(self.secret.encode("utf-8"), msg, hashlib.sha256).digest()
        return base64.b64encode(mac).decode("utf-8")

    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        ts = str(int(time.time()))
        return {
            "CB-ACCESS-KEY": self.key,
            "CB-ACCESS-SIGN": self._sig(ts, method, path, body),
            "CB-ACCESS-TIMESTAMP": ts,
            "Content-Type": "application/json",
        }

    def _get(self, path: str):
        url = self.base_v3 + path
        hdr = self._headers("GET", path)
        r = self.s.get(url, headers=hdr, timeout=self.timeout)
        if r.status_code >= 400:
            logging.error("CB GET %s -> %s body=%s", url, r.status_code, r.text[:600])
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, payload: dict):
        body = json.dumps(payload, separators=(",", ":"))
        url = self.base_v3 + path
        hdr = self._headers("POST", path, body)
        r = self.s.post(url, data=body, headers=hdr, timeout=self.timeout)
        if r.status_code >= 400:
            logging.error("CB POST %s -> %s body=%s", url, r.status_code, r.text[:600])
        r.raise_for_status()
        return r.json()

    def accounts(self):
        return self._get("/accounts")

    def best_bid_ask(self, product_ids: List[str]):
        if not product_ids:
            return []
        q = "?product_ids=" + ",".join(product_ids)
        return self._get("/best_bid_ask" + q).get("pricebooks", [])

    def place_market_order(self, product_id: str, side: str, usd_amount=None, size=None):
        order = {
            "client_order_id": f"ghost-{int(time.time()*1000)}",
            "product_id": product_id,
            "side": side.lower(),
            "order_configuration": {"market_market_ioc": {}},
        }
        oc = order["order_configuration"]["market_market_ioc"]
        if usd_amount is not None:
            oc["quote_size"] = str(usd_amount)
        elif size is not None:
            oc["base_size"] = str(size)
        else:
            raise ValueError("Provide usd_amount or size")
        return self._post("/orders", order)

# =============================== Types/State ===============================
@dataclass
class Bar:
    price: float
    ts: float

@dataclass
class Series:
    prices: deque = field(default_factory=lambda: deque(maxlen=1440))

@dataclass
class Position:
    symbol: str
    qty: float
    avg: float
    peak: float

class Portfolio:
    def __init__(self, equity_usd=START_EQUITY):
        self.cash = equity_usd
        self.positions: Dict[str, Position] = {}
        self.tick = 0
        self.equity_hist: deque = deque(maxlen=200000)
        self.halted = False
        self.last_rebalance_tick = -999
        self.msg = ""
        self.diag = {
            "ranks": [],
            "ticks_left": REBAL_EVERY_TICKS,
            "warmup_left": WARMUP_TICKS,
            "feed_ok": True,
            "feed_via": "sim",
            "last_reason": "",
        }

    def equity(self, prices: Dict[str, float]) -> float:
        v = self.cash
        for p in self.positions.values():
            v += p.qty * prices.get(p.symbol, p.avg)
        return v

    def gross_exposure(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        return 1 - (self.cash / max(eq, 1e-9))

# =============================== Data (SIM + CB overlay) ===============================
series: Dict[str, Series] = {s: Series() for s in UNIVERSE}
latest_prices: Dict[str, float] = {s: 1.0 for s in UNIVERSE}
_last_sim_price = {s: 1.0 for s in UNIVERSE}

cb = None
if EXECUTION_BACKEND == "coinbase" and (CB_API_KEY and CB_API_SECRET):
    try:
        cb = CoinbaseClient(CB_API_KEY, CB_API_SECRET, sandbox=CB_SANDBOX)
    except Exception as e:
        logging.error("Coinbase client init failed: %s", e)
        cb = None

def _sim_snapshot() -> Dict[str, Bar]:
    snap: Dict[str, Bar] = {}
    now = time.time()
    for s in UNIVERSE:
        base = _last_sim_price[s]
        if s == STABLE:
            price = 1.0
        else:
            drift = 0.0008
            noise = random.uniform(-0.004, 0.004)
            price = max(1e-9, base * (1 + drift + noise))
        _last_sim_price[s] = price
        snap[s] = Bar(price=price, ts=now)
    return snap

def get_market_snapshot() -> Dict[str, Bar]:
    snap = _sim_snapshot()
    if cb:
        try:
            books = cb.best_bid_ask(COINBASE_PRODUCTS)
            now = time.time()
            pid_to_sym = {"BTC-USD":"BTC","ETH-USD":"ETH","PEPE-USD":"PEPE","DOGE-USD":"DOGE","SHIB-USD":"SHIB","USDC-USD":"USDC"}
            for b in books:
                pid = b.get("product_id")
                bid = b.get("bid_price")
                ask = b.get("ask_price")
                if not pid or pid not in pid_to_sym: continue
                mid = (float(bid)+float(ask))/2 if bid and ask else None
                if mid:
                    sym = pid_to_sym[pid]
                    if sym in snap:
                        snap[sym].price = mid
                        snap[sym].ts = now
            state.diag["feed_ok"] = True
            state.diag["feed_via"] = "coinbase"
        except Exception as e:
            state.msg = f"Coinbase feed error: {e}"
            state.diag["feed_ok"] = False
            state.diag["feed_via"] = "sim"
    else:
        state.diag["feed_ok"] = True
        state.diag["feed_via"] = "sim"
    return snap

# =============================== Signals (very simple) ===============================
def stdev(vals: List[float]) -> float:
    n=len(vals)
    if n<2: return 0.0
    m=sum(vals)/n
    return (sum((x-m)**2 for x in vals)/(n-1))**0.5

def pct_ret(a, b): 
    return a/b - 1 if b>0 else 0.0

def rank_asset(sym: str) -> float:
    if sym == STABLE: return -1.0
    p = list(series[sym].prices)
    if len(p) < WARMUP_TICKS: return -1.0
    # Momentum + quality / vol
    def mom(win):
        if len(p) < win+2: return 0.0
        a, b = p[-win-1], p[-2]
        return pct_ret(b, a)
    mom1  = mom(12)
    mom4  = mom(48)
    mom24 = mom(288)
    vol   = (stdev([pct_ret(p[i], p[i-1]) for i in range(1, len(p))]) or 1e-9)
    score = (0.5*mom24 + 0.3*mom4 + 0.2*mom1) / max(vol, 1e-9)
    return score

def should_enter(sym: str, score: float) -> bool:
    if ADVISOR_MODE == "conservative":
        return score >= (MIN_SCORE_ENTER + 0.25)
    return score >= MIN_SCORE_ENTER

# =============================== Core Step ===============================
def rebalance(port: Portfolio, snap: Dict[str, Bar]):
    # append prices
    for sym, bar in snap.items():
        series[sym].prices.append(bar.price)

    prices = {k: v.price for k, v in snap.items()}
    latest_prices.update(prices)

    # warmup
    nonstable_lengths = [len(series[s].prices) for s in UNIVERSE if s != STABLE]
    port.diag["warmup_left"] = max(0, WARMUP_TICKS - (min(nonstable_lengths) if nonstable_lengths else 0))

    # cadence gate
    ticks_since = port.tick - port.last_rebalance_tick
    port.diag["ticks_left"] = max(0, REBAL_EVERY_TICKS - ticks_since)
    if ticks_since < REBAL_EVERY_TICKS:
        return
    port.last_rebalance_tick = port.tick

    # rank
    scores = {sym: rank_asset(sym) for sym in UNIVERSE}
    brain = [{"sym": s, "score": round(scores.get(s, -1.0), 4)} for s in UNIVERSE]
    brain.sort(key=lambda x: x["score"], reverse=True)
    port.diag["ranks"] = brain[:8]

    # keep current positions if score ok
    kept, exits = [], []
    for sym, pos in list(port.positions.items()):
        sc = scores.get(sym, -1.0)
        if sc >= MIN_SCORE_EXIT: kept.append(sym)
        else: exits.append(sym)
    for sym in exits:
        pos = port.positions.get(sym)
        if not pos: continue
        port.cash += pos.qty * prices.get(sym, pos.avg)
        del port.positions[sym]

    ranked = [s for s in sorted(UNIVERSE, key=lambda x: scores.get(x, -1.0), reverse=True)
              if s != STABLE and scores.get(s, -1.0) >= MIN_SCORE_ENTER]
    targets = (kept + [s for s in ranked if s not in kept])[:TOP_N]

    eq = port.equity(prices)
    current_gross = port.gross_exposure(prices)
    target_gross = MAX_GROSS_EXPOSURE
    cash_budget = eq * max(0.0, (target_gross - current_gross))
    per_trade_cash = min(eq * MAX_PER_TRADE_FRAC, cash_budget)

    buys=[]
    if targets and per_trade_cash >= MIN_TRADE_USD:
        for sym in targets:
            if sym in port.positions: continue
            if not should_enter(sym, scores[sym]): continue
            qty = per_trade_cash / prices[sym]
            port.cash -= qty * prices[sym]
            port.positions[sym] = Position(sym, qty, prices[sym], prices[sym])
            buys.append(sym)

    reason = []
    if exits: reason.append("EXIT " + ",".join(exits))
    if buys:  reason.append("BUY " + ",".join(buys))
    port.diag["last_reason"] = "; ".join(reason) if reason else "HOLD"

# =============================== Feasibility display ===============================
def required_daily_growth(cur, goal, days):
    try: return (goal/max(1e-9,cur))**(1/max(1,days)) - 1
    except Exception: return 0.0

def progress_line(port: Portfolio, prices: Dict[str, float]) -> str:
    eq = port.equity(prices)
    port.equity_hist.append(eq)
    try:
        today = date.today()
        deadline = date.fromisoformat(GOAL_DEADLINE)
        days_left = max(0, (deadline - today).days)
    except Exception:
        days_left = 0
    req = required_daily_growth(eq, GOAL_EQUITY, max(1, days_left))
    return f"Equity ${eq:,.2f} | req/day {req*100:.2f}% | days_left {days_left}"

# =============================== Server / UI ===============================
app = Flask(__name__)
state = Portfolio(START_EQUITY)

def authed(req): return req.args.get("token") == TOKEN

@app.get("/")
def root(): return redirect(f"/view?token={TOKEN}", code=302)

@app.get("/health")
def health():
    return jsonify({"ok": True, "ts": int(time.time()), "mode": ADVISOR_MODE, "via": state.diag.get("feed_via","sim")})

@app.get("/api/status")
def api_status():
    if not authed(request): return jsonify({"error":"forbidden"}), 403
    prices = dict(latest_prices)
    eq = state.equity(prices)
    payload = {
        "ok": True,
        "equity": eq,
        "target": GOAL_EQUITY,
        "progress_pct": (eq/GOAL_EQUITY if GOAL_EQUITY>0 else 0.0),
        "gross_exposure": state.gross_exposure(prices),
        "halted": state.halted,
        "msg": state.msg,
        "held": [f"{s}:{float(p.qty):.4g}" for s,p in state.positions.items()],
        "coins_held": len(state.positions),
        "stable_pct": (state.cash/max(eq,1e-9)),
        "ticks_left": int(state.diag.get("ticks_left", REBAL_EVERY_TICKS)),
        "warmup_left": int(state.diag.get("warmup_left", WARMUP_TICKS)),
        "brain": state.diag.get("ranks", []),
        "last_reason": state.diag.get("last_reason", ""),
        "ts": int(time.time()),
        "via": state.diag.get("feed_via", "sim"),
        "advisor_mode": ADVISOR_MODE,
        "version": VERSION
    }
    return jsonify(payload)

@app.get("/view")
def view():
    if not authed(request): return Response("Forbidden", status=403)
    html = """<!doctype html><html><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Ghost Mini</title>
<style>
  html,body{margin:0;height:100%;background:#0d6b24;color:#fff;font-family:system-ui,Segoe UI,Roboto,Inter,Arial}
  .wrap{max-width:1100px;margin:0 auto;padding:20px;display:flex;flex-direction:column;gap:16px}
  .card{background:rgba(0,0,0,.12);border:1px solid rgba(255,255,255,.18);border-radius:12px;padding:16px}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px}
  .label{font-size:12px;text-transform:uppercase;letter-spacing:.08em;opacity:.9;margin-bottom:8px}
  .big{font-size:36px;font-weight:800}
  .mono{font-family:ui-monospace,Menlo,Consolas,monospace}
  .bar{height:12px;background:rgba(255,255,255,.18);border-radius:999px;overflow:hidden}
  .bar>span{display:block;height:100%;background:#fff}
  .chip{display:inline-block;background:rgba(0,0,0,.18);border:1px solid rgba(255,255,255,.18);padding:6px 10px;border-radius:999px;font-size:12px;margin:3px}
  .badge{padding:6px 10px;border-radius:999px;font-weight:800}
  .ok{background:rgba(255,255,255,.22)}
  .fault{background:#ff3b3b;color:#000}
</style></head><body>
<div class="wrap">
  <div class="card">
    <div class="label">Ghost Mini — <span class="mono">v{VER}</span> · <span id="mode" class="mono"></span></div>
    <div id="clock" class="mono" style="opacity:.9"></div>
  </div>
  <div class="grid">
    <div class="card">
      <div class="label">Total Equity</div>
      <div class="big mono" id="equity">$0</div>
      <div>Target <span class="mono" id="target">$0</span></div>
      <div style="height:10px"></div>
      <div class="bar"><span id="progress" style="width:0%"></span></div>
    </div>
    <div class="card">
      <div class="label">Engine</div>
      <div class="mono" id="via">via —</div>
      <div class="mono" id="reason" style="margin-top:6px">—</div>
    </div>
    <div class="card">
      <div class="label">Wallet</div>
      <div id="held" class="mono">—</div>
    </div>
    <div class="card">
      <div class="label">Top Scores</div>
      <div id="tops" class="mono">—</div>
    </div>
  </div>
</div>
<script>
const POLL={POLL};
function chip(x){return `<span class='chip'>${x}</span>`}
function fmt(x){return "$"+Number(x).toLocaleString(undefined,{maximumFractionDigits:6})}
function clamp(v,a,b){return Math.max(a,Math.min(b,v))}
async function pull(){
  try{
    const r = await fetch("/api/status?token={TOKEN}",{cache:"no-store"});
    const s = await r.json();
    document.getElementById("clock").textContent=(new Date()).toUTCString();
    document.getElementById("mode").textContent="mode: "+(s.advisor_mode||"-")+" · "+(s.version||"");
    document.getElementById("via").textContent="feed: "+(s.via||"-");
    document.getElementById("equity").textContent=fmt(s.equity||0);
    document.getElementById("target").textContent=fmt(s.target||0);
    document.getElementById("reason").textContent=s.last_reason||"—";
    const prog=clamp((s.progress_pct||0),0,1);
    document.getElementById("progress").style.width=(prog*100).toFixed(1)+"%";
    const held=(s.held||[]).map(chip).join("")||"—";
    document.getElementById("held").innerHTML=held;
    const tops=(s.brain||[]).map(b=>`${b.sym||b.SYM||'-'}:${b.score}`).join("  ")||"—";
    document.getElementById("tops").textContent=tops;
  }catch(e){ /* ignore */ }
}
pull(); setInterval(pull, POLL);
</script>
</body></html>
""".replace("{POLL}", str(POLL_SEC*1000)).replace("{TOKEN}", TOKEN).replace("{VER}", VERSION)
    return Response(html, mimetype="text/html")

# =============================== Runner ===============================
def run_server():
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logging.info("=== Ghost Mini %s START === Paper=%s Loop=%ss Exec=%s CB=%s", VERSION, PAPER_TRADE, LOOP_SECONDS, EXECUTION_BACKEND, "on" if cb else "off")
    threading.Thread(target=run_server, daemon=True).start()

    while True:
        snap = get_market_snapshot()
        rebalance(state, snap)
        prices = {k: v.price for k, v in snap.items()}
        held = ", ".join([f"{s}:{state.positions[s].qty:.4g}" for s in state.positions]) or "(none)"
        logging.info("[Tick %d] %s | Held: %s | via=%s", state.tick, progress_line(state, prices), held, state.diag.get("feed_via"))
        state.msg = ""
        state.tick += 1
        time.sleep(LOOP_SECONDS)

if __name__ == "__main__":
    main()

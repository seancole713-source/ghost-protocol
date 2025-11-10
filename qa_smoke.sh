#!/usr/bin/env bash
# qa_smoke.sh — Ghost Protocol cockpit end-to-end smoke test
# Quote-only. No on-chain execution.
set -euo pipefail

HOST_URL="${HOST_URL:-http://127.0.0.1:5000}"
TOKEN="${GHOST_API_TOKEN:-}"
OUT_DIR="./qa_out"
TS="$(date -u +"%Y%m%dT%H%M%SZ")"
RUN_DIR="$OUT_DIR/$TS"
mkdir -p "$RUN_DIR"

red(){ printf "\033[31m%s\033[0m\n" "$*"; }
green(){ printf "\033[32m%s\033[0m\n" "$*"; }
yellow(){ printf "\033[33m%s\033[0m\n" "$*"; }
blue(){ printf "\033[36m%s\033[0m\n" "$*"; }

need() { command -v "$1" >/dev/null 2>&1 || { red "Missing dependency: $1"; exit 2; }; }
need curl

PRETTY=""
if command -v jq >/dev/null 2>&1; then
  PRETTY="jq"
elif command -v python3 >/dev/null 2>&1; then
  PRETTY="python3 -m json.tool"
fi

if [[ -z "$TOKEN" ]]; then
  red "GHOST_API_TOKEN not set. Export it and rerun."
  exit 1
fi

HDR=(-H "Authorization: Bearer ${TOKEN}" -H "Content-Type: application/json")
PASS=0; FAIL=0
RES_JSON="$RUN_DIR/summary.jsonl"
touch "$RES_JSON"

save_result(){ # name http size note
  local name="$1" http="$2" size="$3" note="${4:-}"
  if [[ "$http" == "200" ]]; then PASS=$((PASS+1)); ok=1; else FAIL=$((FAIL+1)); ok=0; fi
  printf '{"ts":"%s","name":"%s","http":%s,"size":%s,"ok":%s,"note":%s}\n' \
    "$TS" "$name" "$http" "$size" "$ok" "$(printf '%s' "${note//\"/\\\"}" | sed 's/.*/"&"/')" >> "$RES_JSON"
  if [[ "$ok" == 1 ]]; then green "✓ $name (HTTP $http, $size bytes)"; else red "✗ $name (HTTP $http, $size bytes) $note"; fi
}

check(){
  local ep="$1" file="$RUN_DIR/$(echo "$1" | tr '/\"' '__').json"
  local code size
  code=$(curl -sS "${HDR[@]}" -o "$file" -w "%{http_code}" "$HOST_URL/$ep" || echo "000")
  size=$(wc -c < "$file" 2>/dev/null || echo 0)
  save_result "/$ep" "$code" "$size"
}

post(){
  local ep="$1" body="$2" file="$RUN_DIR/POST_$(echo "$1" | tr '/\"' '__').json"
  local code size
  code=$(curl -sS "${HDR[@]}" -d "$body" -o "$file" -w "%{http_code}" "$HOST_URL/$ep" || echo "000")
  size=$(wc -c < "$file" 2>/dev/null || echo 0)
  save_result "POST /$ep" "$code" "$size"
}

blue "Ghost QA Smoke — $HOST_URL"
echo "Output -> $RUN_DIR"

# 0) Health
check "health"

# 1) Refresh all + source status
check "refresh/all"
check "source/status"

# 2) Goals save + compute
post "goals" '{"daily":12,"weekly":70,"monthly":300,"yearly":3600}'
check "goals?horizon=daily"

# 3) Risk save + get
post "risk" '{"slippage_bps":50,"stop_loss_pct":0.6,"trailing_stop_pct":0.12}'
check "risk"

# 4) Presales add + list
post "presales" '{"project":"LILPEPE","chain":"ETH","stage":"presale"}'
check "presales"

# 5) Wallets + portfolio (addresses provided by user)
post "wallets" '{"evm":{"1":["0x4f33f5e4322e2c8ff95159e2eae8057190217ac7","0xccB365D2e11aE4D6d74715c680f56cf58bF4bF10"]}}'
check "portfolio"

# 6) Stocks lots + read (example)
post "stocks/holdings" '{"AAPL":10,"MSFT":5}'
check "stocks"
check "stocks/holdings"

# 7) Core data
check "ghostscore"
check "fusionai"
check "news"
check "whales"
check "gamestats"
check "advisor/enhanced"
check "diagnostics"

# 8) Quote-only (no execution)
post "trade/quote" '{"sell":"USDC","buy":"ETH","amount":5,"slippage_bps":50}' || true

# Pretty print key artifacts
for f in source__status.json fusionai.json advisor__enhanced.json portfolio.json stocks.json diagnostics.json POST_trade__quote.json; do
  p="$RUN_DIR/$f"
  [[ -s "$p" ]] || continue
  echo -e "\n----- $f -----" >> "$RUN_DIR/pretty.txt"
  if [[ -n "$PRETTY" ]]; then cat "$p" | eval "$PRETTY" >> "$RUN_DIR/pretty.txt" || cat "$p" >> "$RUN_DIR/pretty.txt"; else cat "$p" >> "$RUN_DIR/pretty.txt"; fi
done

# Badge summary
BADGES="unknown"
if [[ -s "$RUN_DIR/source__status.json" ]]; then
  if command -v jq >/dev/null 2>&1; then
    BADGES=$(jq -r '.sources | to_entries | map("\(.key)=\(.value.ok)") | join(", ")' "$RUN_DIR/source__status.json")
  fi
fi

echo
blue "Badge status: $BADGES"
if [[ -s "$RUN_DIR/POST_trade__quote.json" ]]; then
  if command -v jq >/dev/null 2>&1; then
    SRC=$(jq -r '.source // "n/a"' "$RUN_DIR/POST_trade__quote.json")
    PRICE=$(jq -r '.price // .amountOut // "n/a"' "$RUN_DIR/POST_trade__quote.json")
    echo "Quote source: $SRC  price/amountOut: $PRICE"
  fi
fi

echo
if [[ $FAIL -eq 0 ]]; then
  green "SMOKE PASS — $PASS checks green"
  exit 0
else
  yellow "SMOKE PARTIAL — $PASS pass / $FAIL fail. See $RUN_DIR for logs."
  exit 1
fi
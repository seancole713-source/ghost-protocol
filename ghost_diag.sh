# ghost_diag.sh
# Ghost Protocol — Comprehensive diagnostics (advisory-safe)
# Requirements: env GHOST_API_TOKEN set; optional HOST_URL (default http://127.0.0.1:5000); jq installed.

#!/usr/bin/env bash
set -Eeuo pipefail

# --- Config & guards ---
HOST_URL="${HOST_URL:-http://127.0.0.1:5000}"
: "${GHOST_API_TOKEN:?GHOST_API_TOKEN is not set}"
USE_PYTHON_JSON=false
command -v jq >/dev/null || USE_PYTHON_JSON=true

AUTH_HEADER=("Authorization: Bearer $GHOST_API_TOKEN")
JSON_HEADER=("Content-Type: application/json")
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="diag_out/$TS"
mkdir -p "$OUT_DIR"

ce() {  # curl GET -> stdout
  curl -sS -H "${AUTH_HEADER[0]}" "$HOST_URL$1"
}
cpj() { # curl POST JSON -> stdout
  curl -sS -H "${AUTH_HEADER[0]}" -H "${JSON_HEADER[0]}" -d "$2" "$HOST_URL$1"
}

head_bytes() { # $1 content, $2 N, $3 file
  printf "%s" "$1" | head -c "$2" > "$3"
}

# --- Begin run ---
echo "== Ghost Diagnostics @ $TS =="
echo "HOST_URL=$HOST_URL"
echo

# 0) Optional health ping (ignore errors)
HEALTH_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$HOST_URL/health" || true)
echo "[health] http=$HEALTH_CODE"

# 1) Refresh all
REFRESH_JSON="$(ce /refresh/all || echo '{}')"
printf "%s\n" "$REFRESH_JSON" > "$OUT_DIR/refresh_all.json"

# 2) Source status (save 200B artifact)
STATUS_JSON="$(ce /source/status || echo '{}')"
printf "%s\n" "$STATUS_JSON" > "$OUT_DIR/source_status.json"
head_bytes "$STATUS_JSON" 200 "$OUT_DIR/source_status_200b.txt"

if [ "$USE_PYTHON_JSON" = "true" ]; then
  OK_SOURCES=$(printf "%s" "$STATUS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for s in d.get('sources',{}).values() if s.get('ok',False)))" 2>/dev/null || echo 0)
  ALL_SOURCES=$(printf "%s" "$STATUS_JSON" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('sources',{})))" 2>/dev/null || echo 0)
else
  OK_SOURCES=$(printf "%s" "$STATUS_JSON" | jq '[.sources|to_entries[]|select(.value.ok==true)]|length' 2>/dev/null || echo 0)
  ALL_SOURCES=$(printf "%s" "$STATUS_JSON" | jq '.sources|length' 2>/dev/null || echo 0)
fi
echo "[sources] $OK_SOURCES/$ALL_SOURCES ok"

# 3) Ghost Brain
GHOST_JSON="$(ce /ghostscore || echo '{}')"
printf "%s\n" "$GHOST_JSON" > "$OUT_DIR/ghostscore.json"
GHOST_SUMMARY=$(printf "%s" "$GHOST_JSON" | jq -c '{score, regime, btc_change, eth_change, xrp_change}' 2>/dev/null || echo '{}')
echo "[ghost] $GHOST_SUMMARY"

# 4) Fusion AI
FUSION_JSON="$(ce /fusionai || echo '{}')"
printf "%s\n" "$FUSION_JSON" > "$OUT_DIR/fusionai.json"
FUSION_ROWS=$(printf "%s" "$FUSION_JSON" | jq '(.rows // [])|length' 2>/dev/null || echo 0)
FUSION_BUY=$(printf "%s" "$FUSION_JSON" | jq '[(.rows // [])[]|select((.action|tostring|ascii_downcase)=="buy")]|length' 2>/dev/null || echo 0)
FUSION_HOLD=$(printf "%s" "$FUSION_JSON" | jq '[(.rows // [])[]|select((.action|tostring|ascii_downcase)=="hold")]|length' 2>/dev/null || echo 0)
FUSION_SELL=$(printf "%s" "$FUSION_JSON" | jq '[(.rows // [])[]|select((.action|tostring|ascii_downcase)=="sell")]|length' 2>/dev/null || echo 0)
echo "[fusion] rows=$FUSION_ROWS buy=$FUSION_BUY hold=$FUSION_HOLD sell=$FUSION_SELL"

# 5) Advisor decisions
ADV_JSON="$(ce /advisor/enhanced || echo '{}')"
printf "%s\n" "$ADV_JSON" > "$OUT_DIR/advisor_enhanced.json"
ADV_COUNT=$(printf "%s" "$ADV_JSON" | jq '(.decisions // [])|length' 2>/dev/null || echo 0)
echo "[advisor] decisions=$ADV_COUNT"

# 6) Goals (daily horizon)
GOALS_JSON="$(ce '/goals?horizon=daily' || echo '{}')"
printf "%s\n" "$GOALS_JSON" > "$OUT_DIR/goals_daily.json"
GOALS_SAVED=$(printf "%s" "$GOALS_JSON" | jq 'has("targets")' 2>/dev/null || echo false)
GOALS_COMPUTED=$(printf "%s" "$GOALS_JSON" | jq '((.path // [])|length>0)' 2>/dev/null || echo false)
echo "[goals] saved=$GOALS_SAVED computed=$GOALS_COMPUTED"

# 7) Risk
RISK_JSON="$(ce /risk || echo '{}')"
printf "%s\n" "$RISK_JSON" > "$OUT_DIR/risk.json"
RISK_SUMMARY=$(printf "%s" "$RISK_JSON" | jq -c '{slippage_bps, stop_loss_pct, trailing_stop_pct}' 2>/dev/null || echo '{}')
echo "[risk] $RISK_SUMMARY"

# 8) Portfolio (save 200B artifact too)
PORT_JSON="$(ce /portfolio || echo '{}')"
printf "%s\n" "$PORT_JSON" > "$OUT_DIR/portfolio.json"
head_bytes "$PORT_JSON" 200 "$OUT_DIR/portfolio_200b.txt"
PORT_TOTAL=$(printf "%s" "$PORT_JSON" | jq '.total_usd // .total // 0' 2>/dev/null || echo 0)
PORT_ERRORS=$(printf "%s" "$PORT_JSON" | jq '(.errors // [])|length' 2>/dev/null || echo 0)
echo "[portfolio] total_usd=$PORT_TOTAL errors=$PORT_ERRORS"
# top5 by value (if present)
printf "%s" "$PORT_JSON" | jq -r '
  (.holdings // []) 
  | sort_by(.valueUsd // .value // 0) | reverse 
  | .[0:5] 
  | map({symbol:(.symbol // .token // "NA"), qty:(.qty // .amount // 0), value:(.valueUsd // .value // 0)})' > "$OUT_DIR/portfolio_top5.json" 2>/dev/null || true

# 9) Stocks
STOCKS_JSON="$(ce /stocks || echo '[]')"
printf "%s\n" "$STOCKS_JSON" > "$OUT_DIR/stocks.json"
STOCKS_COUNT=$(printf "%s" "$STOCKS_JSON" | jq 'length' 2>/dev/null || echo 0)
STOCKS_PRICED=$(printf "%s" "$STOCKS_JSON" | jq '[.[]|select((.price // 0) > 0)]|length' 2>/dev/null || echo 0)
echo "[stocks] items=$STOCKS_COUNT priced_nonzero=$STOCKS_PRICED"

# 10) Presales
PRESALES_JSON="$(ce /presales || echo '{}')"
printf "%s\n" "$PRESALES_JSON" > "$OUT_DIR/presales.json"
PRESALES_COUNT=$(printf "%s" "$PRESALES_JSON" | jq '(if has("items") then .items else . end | length)' 2>/dev/null || echo 0)
echo "[presales] count=$PRESALES_COUNT"

# 11) News
NEWS_JSON="$(ce /news || echo '{}')"
printf "%s\n" "$NEWS_JSON" > "$OUT_DIR/news.json"
NEWS_COUNT=$(printf "%s" "$NEWS_JSON" | jq '(.items // [])|length' 2>/dev/null || echo 0)
echo "[news] headlines=$NEWS_COUNT"

# 12) Game stats
GAME_JSON="$(ce /gamestats || echo '{}')"
printf "%s\n" "$GAME_JSON" > "$OUT_DIR/gamestats.json"
GAME_SUMMARY=$(printf "%s" "$GAME_JSON" | jq -c '{level, trades, win_rate_pct}' 2>/dev/null || echo '{}')
echo "[gamestats] $GAME_SUMMARY"

# 13) Diagnostics (save 200B artifact)
DIAG_JSON="$(ce /diagnostics || echo '{}')"
printf "%s\n" "$DIAG_JSON" > "$OUT_DIR/diagnostics.json"
head_bytes "$DIAG_JSON" 200 "$OUT_DIR/diagnostics_200b.txt"
DIAG_OK=$(printf "%s" "$DIAG_JSON" | jq '.ok' 2>/dev/null || echo false)
TRADER_READY=$(printf "%s" "$DIAG_JSON" | jq '.checks.trader_ready // false' 2>/dev/null || echo false)
echo "[diagnostics] ok=$DIAG_OK trader_ready=$TRADER_READY"

# 14) RPC usage (optional endpoint)
RPC_JSON="$(ce /rpc/usage || echo '{}')"
printf "%s\n" "$RPC_JSON" > "$OUT_DIR/rpc_usage.json"
RPC_SUMMARY=$(printf "%s" "$RPC_JSON" | jq -c '{provider:(.provider // .current_provider // null), calls:(.calls_24h // .total_requests // 0), budget_daily:(.budget_daily // 0)}' 2>/dev/null || echo '{}')
echo "[rpc] $RPC_SUMMARY"

# 15) Quote-only sanity (no execution) — save 1KB artifact
QUOTE_JSON="$(cpj /trade/quote '{"sell":"USDC","buy":"ETH","amount":5,"slippage_bps":50}' || echo '{}')"
printf "%s\n" "$QUOTE_JSON" > "$OUT_DIR/trade_quote.json"
head_bytes "$QUOTE_JSON" 1024 "$OUT_DIR/trade_quote_1kb.json"
QUOTE_SUMMARY=$(printf "%s" "$QUOTE_JSON" | jq -c '{ok, source, route, price, amountOut, router}' 2>/dev/null || echo '{}')
echo "[quote] $QUOTE_SUMMARY"

# 16) Build summary.json
jq -n \
  --argjson badges "$(printf "%s" "$STATUS_JSON" | jq '{coins:.sources.coins.ok, fusion:.sources.fusion.ok, ghost:.sources.ghost.ok, stocks:.sources.stocks.ok, whales:.sources.whales.ok, news:.sources.news.ok, portfolio:.sources.portfolio.ok, presales:.sources.presales.ok, goals:.sources.goals.ok, risk:.sources.risk.ok}' 2>/dev/null || echo '{}')" \
  --argjson fusion "$(jq -n --arg rows "$FUSION_ROWS" --arg buy "$FUSION_BUY" --arg hold "$FUSION_HOLD" --arg sell "$FUSION_SELL" '{rows:($rows|tonumber),buy:($buy|tonumber),hold:($hold|tonumber),sell:($sell|tonumber)}')" \
  --argjson advisor "$(jq -n --arg d "$ADV_COUNT" '{decisions:($d|tonumber)}')" \
  --argjson goals "$(jq -n --arg s "$GOALS_SAVED" --arg c "$GOALS_COMPUTED" '{saved:($s=="true"),computed:($c=="true")}')" \
  --argjson risk "$RISK_SUMMARY" \
  --argjson portfolio "$(jq -n --arg total "$PORT_TOTAL" --arg errs "$PORT_ERRORS" '{total_usd:($total|tonumber),errors:($errs|tonumber)}')" \
  --argjson stocks "$(jq -n --arg cnt "$STOCKS_COUNT" --arg priced "$STOCKS_PRICED" '{items:($cnt|tonumber),priced_nonzero:($priced|tonumber)}')" \
  --arg presales_count "$PRESALES_COUNT" \
  --arg news_count "$NEWS_COUNT" \
  --argjson gamestats "$GAME_SUMMARY" \
  --argjson quote "$QUOTE_SUMMARY" \
  --argjson rpc "$RPC_SUMMARY" \
  --arg ok_sources "$OK_SOURCES" --arg all_sources "$ALL_SOURCES" \
  '{
     meta:{ts:"'"$TS"'", host:"'"$HOST_URL"'" , sources_ok:($ok_sources|tonumber), sources_total:($all_sources|tonumber)},
     badges:$badges,
     fusion:$fusion,
     advisor:$advisor,
     goals:$goals,
     risk:$risk,
     portfolio:$portfolio,
     stocks:$stocks,
     presales:{count:($presales_count|tonumber)},
     news:{headlines:($news_count|tonumber)},
     gamestats:$gamestats,
     quote:$quote,
     rpc_usage:$rpc
   }' | tee "$OUT_DIR/summary.json" >/dev/null

# 17) Pretty summary
{
  echo "=== SUMMARY ==="
  jq -r '.meta, .badges, .fusion, .advisor, .goals, .risk, .portfolio, .stocks, .presales, .news, .gamestats, .quote, .rpc_usage' "$OUT_DIR/summary.json"
  echo
  echo "Artifacts:"
  echo "  $OUT_DIR/source_status_200b.txt"
  echo "  $OUT_DIR/diagnostics_200b.txt"
  echo "  $OUT_DIR/trade_quote_1kb.json"
} | tee "$OUT_DIR/pretty.txt" >/dev/null

echo "Done. Output saved to $OUT_DIR"
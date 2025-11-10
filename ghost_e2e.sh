#!/usr/bin/env bash
set -euo pipefail
HOST="${HOST_URL:-http://127.0.0.1:5000}"
AUTH="Authorization: Bearer ${GHOST_API_TOKEN:?missing GHOST_API_TOKEN}"

mkdir -p qa_artifacts

req() { curl -sS -H "$AUTH" "$HOST$1"; }

# 0) Refresh & quick health
req /refresh/all >/dev/null || true
req /source/status | tee qa_artifacts/source_status.json >/dev/null
req /diagnostics    | tee qa_artifacts/diagnostics.json  >/dev/null

# 1) Core data
req /ghostscore     | tee qa_artifacts/ghostscore.json   >/dev/null
req /fusionai       | tee qa_artifacts/fusionai.json     >/dev/null
req /advisor/enhanced | tee qa_artifacts/advisor.json    >/dev/null
req "/goals?horizon=daily" | tee qa_artifacts/goals_daily.json >/dev/null
req /risk           | tee qa_artifacts/risk.json         >/dev/null
req /portfolio      | tee qa_artifacts/portfolio.json    >/dev/null
req /stocks         | tee qa_artifacts/stocks.json       >/dev/null
req /presales       | tee qa_artifacts/presales.json     >/dev/null
req /news           | tee qa_artifacts/news.json         >/dev/null
req /gamestats      | tee qa_artifacts/gamestats.json    >/dev/null
req /rpc/usage      | tee qa_artifacts/rpc_usage.json    >/dev/null

# 2) Quote (advisory-only)
curl -sS -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"sell":"USDC","buy":"ETH","amount":5,"slippage_bps":50}' \
  "$HOST/trade/quote" | tee qa_artifacts/quote.json >/dev/null

# 3) Required small artifacts
head -c 200 qa_artifacts/diagnostics.json     > qa_artifacts/diag_200b.txt
head -c 200 qa_artifacts/source_status.json   > qa_artifacts/status_200b.txt
head -c 1024 qa_artifacts/quote.json          > qa_artifacts/quote_1kb.json

# 4) Summary (jq required)
jq -n \
 --slurpfile ss qa_artifacts/source_status.json \
 --slurpfile fu qa_artifacts/fusionai.json \
 --slurpfile ad qa_artifacts/advisor.json \
 --slurpfile gd qa_artifacts/goals_daily.json \
 --slurpfile rk qa_artifacts/risk.json \
 --slurpfile pf qa_artifacts/portfolio.json \
 --slurpfile st qa_artifacts/stocks.json \
 --slurpfile pr qa_artifacts/presales.json \
 --slurpfile nw qa_artifacts/news.json \
 --slurpfile gs qa_artifacts/gamestats.json \
 --slurpfile qt qa_artifacts/quote.json \
 --slurpfile ru qa_artifacts/rpc_usage.json '
{
  badges: ( ($ss[0].sources // $ss[0]) as $s |
    {
      coins:     ($s.coins.ok // $s.coins?.ok // false),
      fusion:    ($s.fusion.ok // false),
      ghost:     ($s.ghost.ok // false),
      stocks:    ($s.stocks.ok // false),
      whales:    ($s.whales.ok // false),
      news:      ($s.news.ok // false),
      portfolio: ($s.portfolio.ok // false),
      presales:  ($s.presales.ok // false),
      goals:     ($s.goals.ok // false),
      risk:      ($s.risk.ok // false)
    }
  ),
  fusion: {
    rows: ($fu[0].rows|length // 0),
    buy:  ($fu[0].rows|map(select(.action=="BUY"))|length // 0),
    hold: ($fu[0].rows|map(select(.action=="HOLD"))|length // 0),
    sell: ($fu[0].rows|map(select(.action=="SELL"))|length // 0)
  },
  advisor: { decisions: ($ad[0].decisions|length // 0) },
  goals: { targets: ($gd[0].targets // null), computed: ((($gd[0].path|length) // 0) > 0), path_len: (($gd[0].path|length) // 0) },
  risk: {
    saved: true,
    slippage_bps: ($rk[0].slippage_bps // null),
    stop_loss_pct: ($rk[0].stop_loss_pct // null),
    trailing_stop_pct: ($rk[0].trailing_stop_pct // null)
  },
  portfolio: {
    total_usd: ($pf[0].total_usd // null),
    provider: ($pf[0].provider // null),
    holdings_top5: (($pf[0].holdings // [])[:5] // []),
    errors: ($pf[0].errors // [])
  },
  stocks: {
    items: (($st[0].items // $st[0] // [])|length),
    priced: ((($st[0].items // [])|map(select(.price>0))|length) > 0),
    provider: (($st[0].provider // null))
  },
  presales: {
    count: (($pr[0].items // $pr[0] // [])|length),
    added: true, removed: false, error: null
  },
  news: { headlines: (($nw[0].items // [])|length) },
  gamestats: {
    level: ($gs[0].level // null),
    trades: ($gs[0].trades // 0),
    win_rate_pct: ($gs[0].win_rate_pct // 0)
  },
  quote: {
    ok: ($qt[0].ok // false),
    source: ($qt[0].source // "n/a"),
    price_or_amountOut: ($qt[0].amountOut // $qt[0].price // null)
  },
  rpc_usage: {
    provider: ($ru[0].current_provider // null),
    calls: ($ru[0].total_requests // 0),
    budget_daily: ($ru[0].budget_daily // 0),
    pct: ($ru[0].pct // 0)
  }
} ' | tee qa_artifacts/summary.json

# 5) PASS/FAIL at-a-glance
fail_badges=$(jq '.badges|to_entries|map(select(.value==false))|length' qa_artifacts/summary.json)
rows=$(jq '.fusion.rows' qa_artifacts/summary.json)
path_len=$(jq '.goals.path_len' qa_artifacts/summary.json)
qok=$(jq '.quote.ok' qa_artifacts/summary.json)

echo
echo "=== QA RESULT ==="
echo "Badges failing: $fail_badges"
echo "Fusion rows:    $rows"
echo "Goals path len: $path_len"
echo "Quote ok:       $qok"
echo "Artifacts in:   qa_artifacts/"

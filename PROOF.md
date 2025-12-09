# PROOF

- Commit SHA: 55323647370ed5ecbc0f4b4e83050cfb8c66956e
- Config Hash: d9ad70cf26cf39050f90281f19f401528fa5471a779da2d014130b9a78d54bfa
- Config: {"CRYPTO_ENABLED": "1", "CRYPTO_SYMBOLS": "", "FUSION_AI_ON": "",

  "MACRO_BRAIN_ON": "", "NEWS_SENTIMENT_ON": "", "SIM_MODE": "0"}

## SLO Snapshot (Last)

```json
{
  "ts": 1760305871,
  "price": {
    "cached_at": 1760305847.9465935,
    "quorum_size": 2,
    "spread": 0.0001804426790707094,
    "provider": "coingecko"
  },
  "cockpit": {
    "as_o": 1760305871,
    "preds_crypto": 1,
    "feeds": {
      "stocks": true,
      "crypto": true,
      "news": true,
      "telegram": true,
      "prices": true
    }
  }
}

```text

## Metrics (tail)

```text

ghost_telegram_test_seconds_bucket{le="0.25"} 0.0
ghost_telegram_test_seconds_bucket{le="0.5"} 0.0
ghost_telegram_test_seconds_bucket{le="0.75"} 0.0
ghost_telegram_test_seconds_bucket{le="1.0"} 0.0
ghost_telegram_test_seconds_bucket{le="2.5"} 0.0
ghost_telegram_test_seconds_bucket{le="5.0"} 0.0
ghost_telegram_test_seconds_bucket{le="7.5"} 0.0
ghost_telegram_test_seconds_bucket{le="10.0"} 0.0
ghost_telegram_test_seconds_bucket{le="+Inf"} 0.0
ghost_telegram_test_seconds_count 0.0
ghost_telegram_test_seconds_sum 0.0

# HELP ghost_telegram_test_seconds_created Latency of building /api/telegram/test card

# TYPE ghost_telegram_test_seconds_created gauge

ghost_telegram_test_seconds_created 1.7603046432416914e+09

# HELP ghost_telegram_test_total Total /api/telegram/test calls by send flag

# TYPE ghost_telegram_test_total counter

# HELP ghost_alert_queue_length Current number of alerts pending in send queue

# TYPE ghost_alert_queue_length gauge

ghost_alert_queue_length 0.0

# HELP ghost_alert_send_retries_total Total alert send retries across all sinks

# TYPE ghost_alert_send_retries_total counter

ghost_alert_send_retries_total 0.0

# HELP ghost_alert_send_retries_created Total alert send retries across all sinks

# TYPE ghost_alert_send_retries_created gauge

ghost_alert_send_retries_created 1.7603046432417428e+09

# HELP ghost_rate_limit_drops_total Total write requests dropped by rate limiter

# TYPE ghost_rate_limit_drops_total counter

ghost_rate_limit_drops_total 0.0

# HELP ghost_rate_limit_drops_created Total write requests dropped by rate limiter

# TYPE ghost_rate_limit_drops_created gauge

ghost_rate_limit_drops_created 1.7603046432417507e+09

# HELP ghost_rate_limit_tokens Current available tokens in write rate limiter bucket

# TYPE ghost_rate_limit_tokens gauge

ghost_rate_limit_tokens 0.0

# HELP ghost_snapshot_aso Epoch timestamp of the latest snapshot served by /api/cockpit

# TYPE ghost_snapshot_aso gauge

ghost_snapshot_aso 1.760305878e+09

# HELP ghost_decision_final_score Latest fused decision score (alpha*price + beta*news)

# TYPE ghost_decision_final_score gauge

ghost_decision_final_score 0.0

# HELP ghost_why_now_count Count of 'Why now' reasons included in the last signal card

# TYPE ghost_why_now_count gauge

ghost_why_now_count 0.0

# HELP ghost_macro_confidence Macro brain confidence for last advisory (0-100)

# TYPE ghost_macro_confidence gauge

ghost_macro_confidence{scenario="bull"} 38.0
ghost_macro_confidence{scenario="base"} 38.0
ghost_macro_confidence{scenario="bear"} 38.0

# HELP ghost_macro_refresh_total Macro brain refresh computations

# TYPE ghost_macro_refresh_total counter

ghost_macro_refresh_total{result="ok"} 6.0

# HELP ghost_macro_refresh_created Macro brain refresh computations

# TYPE ghost_macro_refresh_created gauge

ghost_macro_refresh_created{result="ok"} 1.7603049783152733e+09

# HELP ghost_llm_calls_total Total LLM advisory calls

# TYPE ghost_llm_calls_total counter

# HELP ghost_llm_decisions_total Total LLM decisions by action

# TYPE ghost_llm_decisions_total counter

# HELP ghost_llm_confidence Last LLM advisory confidence (0-100)

# TYPE ghost_llm_confidence gauge

# HELP ghost_predict_runs_total Total prediction runs by symbol

# TYPE ghost_predict_runs_total counter

# HELP ghost_predict_outcomes_total Total prediction outcomes by symbol and hit status

# TYPE ghost_predict_outcomes_total counter

# HELP ghost_predict_mae Mean Absolute Error for predictions

# TYPE ghost_predict_mae gauge

# HELP ghost_predict_mape Mean Absolute Percentage Error for predictions

# TYPE ghost_predict_mape gauge

# HELP ghost_predict_rmse Root Mean Squared Error for predictions

# TYPE ghost_predict_rmse gauge

# HELP ghost_predict_confidence_avg Average prediction confidence

# TYPE ghost_predict_confidence_avg gauge

# HELP ghost_crypto_price_fetch_total Total crypto price fetches

# TYPE ghost_crypto_price_fetch_total counter

ghost_crypto_price_fetch_total{provider="coingecko",result="success"} 4.0

# HELP ghost_crypto_price_fetch_created Total crypto price fetches

# TYPE ghost_crypto_price_fetch_created gauge

ghost_crypto_price_fetch_created{provider="coingecko",result="success"} 1.7603058479467947e+09

# HELP ghost_crypto_predict_seconds Crypto prediction generation duration

# TYPE ghost_crypto_predict_seconds histogram

# HELP ghost_prediction_mape Mean Absolute Percentage Error for predictions

# TYPE ghost_prediction_mape gauge

# HELP ghost_sentiment_score News sentiment score

# TYPE ghost_sentiment_score gauge

# HELP ghost_http_pool_used_total Total HTTP requests performed using pooled sessions

# TYPE ghost_http_pool_used_total counter

ghost_http_pool_used_total{host="api.polygon.io"} 5.0
ghost_http_pool_used_total{host="www.alphavantage.co"} 2.0
ghost_http_pool_used_total{host="query1.finance.yahoo.com"} 2.0

# HELP ghost_http_pool_used_created Total HTTP requests performed using pooled sessions

# TYPE ghost_http_pool_used_created gauge

ghost_http_pool_used_created{host="api.polygon.io"} 1.760304653310114e+09
ghost_http_pool_used_created{host="www.alphavantage.co"} 1.7603049764742599e+09
ghost_http_pool_used_created{host="query1.finance.yahoo.com"} 1.760304976880193e+09

# HELP ghost_http_direct_used_total Total HTTP requests performed using direct requests.*

# TYPE ghost_http_direct_used_total counter

# HELP ghost_ai_memory_requests_total AI memory endpoint requests

# TYPE ghost_ai_memory_requests_total counter

# HELP ghost_ai_memory_latency_seconds Latency for AI memory endpoints

# TYPE ghost_ai_memory_latency_seconds histogram

```text

## Logs (last 10 ndjson records)

```text

{"ts": 1760305847, "file": "/tmp/ghost_parallel.log", "tail": "INFO:     127.0.0.1:37512 - \"GET /cockpit HTTP/1.1\" 404 Not Found\n{\"ts\":\"2025-10-12T21:36:16.473976+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"price_fallback_persistent\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"ghost\",\"symbol\":\"WOLF\",\"price\":31.1,\"age_hours\":0.09263164010312822}\n{\"ts\":\"2025-10-12T21:36:17.693852+00:00\",\"level\":\"warning\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"provider_error\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"ghost\",\"component\":\"provider\",\"provider\":\"yahoo\",\"error\":\"429 Client Error: Too Many Requests for url: <<<<<https://query1.finance.yahoo.com/v7/finance/quote?symbols=WOLF\"}\n{\"ts\":\"2025-10-12T21:36:17.777723+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed>>>>> to get ticker 'WOLF' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:17.839149+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"WOLF: No price data found, symbol may be delisted (period=2d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:17.983226+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'SMH' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.044366+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SMH: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.106123+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'SOXX' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.168424+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SOXX: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.251007+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'QQQ' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.314228+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"QQQ: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\nINFO:     127.0.0.1:37520 - \"GET /api/cockpit HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:57164 - \"GET /cockpit HTTP/1.1\" 404 Not Found\n{\"ts\":\"2025-10-12T21:50:47.555539+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"Crypto providers initialized\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"ghost\"}\n{\"ts\":\"2025-10-12T21:50:47.847302+00:00\",\"level\":\"warning\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Binance fetch failed for BTC: 451 Client Error:  for url: <<<<<https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:50:47.946674+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto>>>>> price quorum for BTC: $114801.00 (2 providers, 0.02% spread, 85% confidence)\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:     127.0.0.1:57174 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:50:55.974396+00:00\",\"level\":\"warning\",\"logger\":\"urllib3.connectionpool\",\"service\":\"ghost-wol\",\"msg\":\"Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError(\\\"HTTPSConnectionPool(host='www.alphavantage.co', port=443): Read timed out. (read timeout=8)\\\")': /query?function=GLOBAL_QUOTE&symbol=WOLF&apikey=3WNNLA81KS7BG4AK\",\"trace_id\":\"bdcb3cf9-3c70-424b-bc71-4150e6f73eac\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"urllib3.connectionpool\"}\nINFO:     127.0.0.1:58654 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:58668 - \"GET /api/cockpit HTTP/1.1\" 200 OK\n"}
{"ts": 1760305847, "file": "/tmp/ghost_check.log", "tail":
"{\"ts\":\"2025-10-12T21:19:25.123744+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for BTC: $115397.00 (2 providers, 0.08% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:19:25.232114+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for ETH: $4153.37 (2 providers, 0.20% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:19:25.342936+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for SOL: $197.27 (2 providers, 0.20% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:
127.0.0.1:56012 - \"GET /api/crypto/watchlist HTTP/1.1\" 200
OK\n{\"ts\":\"2025-10-12T21:19:26.663308+00:00\",\"level\":\"warning\",\"logger\":\"root\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood file not found:
data/market_mood.json\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"root\"}\n{\"ts\":\"2025-10-12T21:19:26.663401+00:00\",\"level\":\"info\",\"logger\":\"core.stage1_integration\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood stale,
updating...\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"core.stage1_integration\"}\n{\"ts\":\"2025-10-12T21:19:26.723632+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed
to get ticker 'SPY' reason: Expecting value: line 1 column 1 (char
0)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:26.784947+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SPY:
No price data found, symbol may be delisted
(period=5d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:27.918520+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"QQQ:
No price data found, symbol may be delisted
(period=5d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:27.979336+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed
to get ticker '^VIX' reason: Expecting value: line 1 column 1 (char
0)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:28.042510+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"^VIX:
No price data found, symbol may be delisted
(period=1d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:28.043260+00:00\",\"level\":\"error\",\"logger\":\"root\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood update failed: Insufficient SPY
data\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"root\"}\nINFO:
127.0.0.1:56026 - \"POST /api/telegram/test HTTP/1.1\" 200 OK\nINFO: 127.0.0.1:47566 - \"GET /cockpit HTTP/1.1\" 404 Not
Found\nINFO: 127.0.0.1:41194 - \"GET /cockpit HTTP/1.1\" 404 Not Found\nINFO: 127.0.0.1:45026 - \"GET /metrics
HTTP/1.1\" 200 OK\nINFO: Shutting down\nINFO: Waiting for application shutdown.\nINFO: Application shutdown
complete.\nINFO: Finished server process [169593]\n"}
{"ts": 1760305847, "file": "/tmp/ghost_final.log", "tail": "{\"ts\":\"2025-10-12T21:00:07.608882+00:00\",\"level\":\"error\",\"logger\":\"asyncio\",\"service\":\"ghost-wol\",\"msg\":\"Task exception was never retrieved\\nfuture: <Task finished name='Task-9' coro=<run_forever() done, defined at /workspaces/GHOST/core/workers/pattern_memory.py:100> exception=PermissionError(13, 'Permission denied')>\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"asyncio\",\"error_type\":\"PermissionError\",\"error\":\"[Errno 13] Permission denied: '/data'\"}\n{\"ts\":\"2025-10-12T21:00:07.608965+00:00\",\"level\":\"error\",\"logger\":\"asyncio\",\"service\":\"ghost-wol\",\"msg\":\"Task exception was never retrieved\\nfuture: <Task finished name='Task-10' coro=<run_forever() done, defined at /workspaces/GHOST/core/workers/reflex_trainer.py:89> exception=PermissionError(13, 'Permission denied')>\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"asyncio\",\"error_type\":\"PermissionError\",\"error\":\"[Errno 13] Permission denied: '/data'\"}\nINFO:     Application startup complete.\nINFO:     Uvicorn running on <<<<<http://0.0.0.0:5001>>>>> (Press CTRL+C to quit)\n{\"ts\":\"2025-10-12T21:00:08.770595+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"request\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"ghost\",\"component\":\"api\",\"status\":200,\"duration_ms\":1.59,\"client\":\"127.0.0.1\"}\n[GHOST INIT] ChatGPT Price Provider: DISABLED ('NoneType' object is not callable)\nINFO:     127.0.0.1:58208 - \"GET /health HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:00:18.821546+00:00\",\"level\":\"error\",\"logger\":\"core.edgar_integration\",\"service\":\"ghost-wol\",\"msg\":\"Error converting ticker WOLF to CIK: 404 Client Error: Not Found for url: <<<<<https://data.sec.gov/files/company_tickers.json\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"core.edgar_integration\"}\n{\"ts\":\"2025-10-12T21:00:18.821673+00:00\",\"level\":\"warning\",\"logger\":\"core.edgar_integration\",\"service\":\"ghost-wol\",\"msg\":\"Could>>>>> not find CIK for WOLF\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"core.edgar_integration\"}\n[48h forecast] Generated: 42 at 1760302819\nINFO:     127.0.0.1:35480 - \"GET /health HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:03:21.690196+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"Crypto providers initialized\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"ghost\"}\n{\"ts\":\"2025-10-12T21:03:21.972498+00:00\",\"level\":\"warning\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Binance fetch failed for BTC: 451 Client Error:  for url: <<<<<https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:03:22.047659+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto>>>>> price quorum for BTC: $115023.26 (2 providers, 0.00% spread, 85% confidence)\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:     127.0.0.1:35490 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:37760 - \"GET /metrics HTTP/1.1\" 200 OK\nINFO:     Shutting down\nINFO:     Waiting for application shutdown.\nINFO:     Application shutdown complete.\nINFO:     Finished server process [161970]\n"}
{"ts": 1760305865, "file": "/tmp/ghost_parallel.log", "tail": "{\"ts\":\"2025-10-12T21:36:17.693852+00:00\",\"level\":\"warning\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"provider_error\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"ghost\",\"component\":\"provider\",\"provider\":\"yahoo\",\"error\":\"429 Client Error: Too Many Requests for url: <<<<<https://query1.finance.yahoo.com/v7/finance/quote?symbols=WOLF\"}\n{\"ts\":\"2025-10-12T21:36:17.777723+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed>>>>> to get ticker 'WOLF' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:17.839149+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"WOLF: No price data found, symbol may be delisted (period=2d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:17.983226+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'SMH' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.044366+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SMH: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.106123+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'SOXX' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.168424+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SOXX: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.251007+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'QQQ' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.314228+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"QQQ: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\nINFO:     127.0.0.1:37520 - \"GET /api/cockpit HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:57164 - \"GET /cockpit HTTP/1.1\" 404 Not Found\n{\"ts\":\"2025-10-12T21:50:47.555539+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"Crypto providers initialized\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"ghost\"}\n{\"ts\":\"2025-10-12T21:50:47.847302+00:00\",\"level\":\"warning\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Binance fetch failed for BTC: 451 Client Error:  for url: <<<<<https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:50:47.946674+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto>>>>> price quorum for BTC: $114801.00 (2 providers, 0.02% spread, 85% confidence)\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:     127.0.0.1:57174 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:50:55.974396+00:00\",\"level\":\"warning\",\"logger\":\"urllib3.connectionpool\",\"service\":\"ghost-wol\",\"msg\":\"Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError(\\\"HTTPSConnectionPool(host='www.alphavantage.co', port=443): Read timed out. (read timeout=8)\\\")': /query?function=GLOBAL_QUOTE&symbol=WOLF&apikey=3WNNLA81KS7BG4AK\",\"trace_id\":\"bdcb3cf9-3c70-424b-bc71-4150e6f73eac\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"urllib3.connectionpool\"}\nINFO:     127.0.0.1:58654 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:58668 - \"GET /api/cockpit HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:55250 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:55266 - \"GET /api/cockpit HTTP/1.1\" 200 OK\n"}
{"ts": 1760305865, "file": "/tmp/ghost_check.log", "tail":
"{\"ts\":\"2025-10-12T21:19:25.123744+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for BTC: $115397.00 (2 providers, 0.08% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:19:25.232114+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for ETH: $4153.37 (2 providers, 0.20% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:19:25.342936+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for SOL: $197.27 (2 providers, 0.20% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:
127.0.0.1:56012 - \"GET /api/crypto/watchlist HTTP/1.1\" 200
OK\n{\"ts\":\"2025-10-12T21:19:26.663308+00:00\",\"level\":\"warning\",\"logger\":\"root\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood file not found:
data/market_mood.json\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"root\"}\n{\"ts\":\"2025-10-12T21:19:26.663401+00:00\",\"level\":\"info\",\"logger\":\"core.stage1_integration\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood stale,
updating...\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"core.stage1_integration\"}\n{\"ts\":\"2025-10-12T21:19:26.723632+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed
to get ticker 'SPY' reason: Expecting value: line 1 column 1 (char
0)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:26.784947+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SPY:
No price data found, symbol may be delisted
(period=5d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:27.918520+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"QQQ:
No price data found, symbol may be delisted
(period=5d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:27.979336+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed
to get ticker '^VIX' reason: Expecting value: line 1 column 1 (char
0)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:28.042510+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"^VIX:
No price data found, symbol may be delisted
(period=1d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:28.043260+00:00\",\"level\":\"error\",\"logger\":\"root\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood update failed: Insufficient SPY
data\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"root\"}\nINFO:
127.0.0.1:56026 - \"POST /api/telegram/test HTTP/1.1\" 200 OK\nINFO: 127.0.0.1:47566 - \"GET /cockpit HTTP/1.1\" 404 Not
Found\nINFO: 127.0.0.1:41194 - \"GET /cockpit HTTP/1.1\" 404 Not Found\nINFO: 127.0.0.1:45026 - \"GET /metrics
HTTP/1.1\" 200 OK\nINFO: Shutting down\nINFO: Waiting for application shutdown.\nINFO: Application shutdown
complete.\nINFO: Finished server process [169593]\n"}
{"ts": 1760305865, "file": "/tmp/ghost_final.log", "tail": "{\"ts\":\"2025-10-12T21:00:07.608882+00:00\",\"level\":\"error\",\"logger\":\"asyncio\",\"service\":\"ghost-wol\",\"msg\":\"Task exception was never retrieved\\nfuture: <Task finished name='Task-9' coro=<run_forever() done, defined at /workspaces/GHOST/core/workers/pattern_memory.py:100> exception=PermissionError(13, 'Permission denied')>\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"asyncio\",\"error_type\":\"PermissionError\",\"error\":\"[Errno 13] Permission denied: '/data'\"}\n{\"ts\":\"2025-10-12T21:00:07.608965+00:00\",\"level\":\"error\",\"logger\":\"asyncio\",\"service\":\"ghost-wol\",\"msg\":\"Task exception was never retrieved\\nfuture: <Task finished name='Task-10' coro=<run_forever() done, defined at /workspaces/GHOST/core/workers/reflex_trainer.py:89> exception=PermissionError(13, 'Permission denied')>\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"asyncio\",\"error_type\":\"PermissionError\",\"error\":\"[Errno 13] Permission denied: '/data'\"}\nINFO:     Application startup complete.\nINFO:     Uvicorn running on <<<<<http://0.0.0.0:5001>>>>> (Press CTRL+C to quit)\n{\"ts\":\"2025-10-12T21:00:08.770595+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"request\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"ghost\",\"component\":\"api\",\"status\":200,\"duration_ms\":1.59,\"client\":\"127.0.0.1\"}\n[GHOST INIT] ChatGPT Price Provider: DISABLED ('NoneType' object is not callable)\nINFO:     127.0.0.1:58208 - \"GET /health HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:00:18.821546+00:00\",\"level\":\"error\",\"logger\":\"core.edgar_integration\",\"service\":\"ghost-wol\",\"msg\":\"Error converting ticker WOLF to CIK: 404 Client Error: Not Found for url: <<<<<https://data.sec.gov/files/company_tickers.json\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"core.edgar_integration\"}\n{\"ts\":\"2025-10-12T21:00:18.821673+00:00\",\"level\":\"warning\",\"logger\":\"core.edgar_integration\",\"service\":\"ghost-wol\",\"msg\":\"Could>>>>> not find CIK for WOLF\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"core.edgar_integration\"}\n[48h forecast] Generated: 42 at 1760302819\nINFO:     127.0.0.1:35480 - \"GET /health HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:03:21.690196+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"Crypto providers initialized\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"ghost\"}\n{\"ts\":\"2025-10-12T21:03:21.972498+00:00\",\"level\":\"warning\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Binance fetch failed for BTC: 451 Client Error:  for url: <<<<<https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:03:22.047659+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto>>>>> price quorum for BTC: $115023.26 (2 providers, 0.00% spread, 85% confidence)\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:     127.0.0.1:35490 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:37760 - \"GET /metrics HTTP/1.1\" 200 OK\nINFO:     Shutting down\nINFO:     Waiting for application shutdown.\nINFO:     Application shutdown complete.\nINFO:     Finished server process [161970]\n"}
{"ts": 1760305871, "file": "/tmp/ghost_parallel.log", "tail": "{\"ts\":\"2025-10-12T21:36:17.839149+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"WOLF: No price data found, symbol may be delisted (period=2d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:17.983226+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'SMH' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.044366+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SMH: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.106123+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'SOXX' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.168424+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SOXX: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.251007+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed to get ticker 'QQQ' reason: Expecting value: line 1 column 1 (char 0)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:36:18.314228+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"QQQ: No price data found, symbol may be delisted (period=20d)\",\"trace_id\":\"f8f4caa0-dc7c-482e-92e0-d7cb98025b0a\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"yfinance\"}\nINFO:     127.0.0.1:37520 - \"GET /api/cockpit HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:57164 - \"GET /cockpit HTTP/1.1\" 404 Not Found\n{\"ts\":\"2025-10-12T21:50:47.555539+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"Crypto providers initialized\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"ghost\"}\n{\"ts\":\"2025-10-12T21:50:47.847302+00:00\",\"level\":\"warning\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Binance fetch failed for BTC: 451 Client Error:  for url: <<<<<https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:50:47.946674+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto>>>>> price quorum for BTC: $114801.00 (2 providers, 0.02% spread, 85% confidence)\",\"trace_id\":\"9d250705-53a9-4911-bd87-231adeca687e\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:     127.0.0.1:57174 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:50:55.974396+00:00\",\"level\":\"warning\",\"logger\":\"urllib3.connectionpool\",\"service\":\"ghost-wol\",\"msg\":\"Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError(\\\"HTTPSConnectionPool(host='www.alphavantage.co', port=443): Read timed out. (read timeout=8)\\\")': /query?function=GLOBAL_QUOTE&symbol=WOLF&apikey=3WNNLA81KS7BG4AK\",\"trace_id\":\"bdcb3cf9-3c70-424b-bc71-4150e6f73eac\",\"path\":\"/api/cockpit\",\"method\":\"GET\",\"name\":\"urllib3.connectionpool\"}\nINFO:     127.0.0.1:58654 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:58668 - \"GET /api/cockpit HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:55250 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:55266 - \"GET /api/cockpit HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:55282 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:55286 - \"GET /api/cockpit HTTP/1.1\" 200 OK\n"}
{"ts": 1760305871, "file": "/tmp/ghost_check.log", "tail":
"{\"ts\":\"2025-10-12T21:19:25.123744+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for BTC: $115397.00 (2 providers, 0.08% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:19:25.232114+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for ETH: $4153.37 (2 providers, 0.20% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:19:25.342936+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto
price quorum for SOL: $197.27 (2 providers, 0.20% spread, 85%
confidence)\",\"trace_id\":\"556c83ed-42aa-49e3-a503-f15cf0374a0a\",\"path\":\"/api/crypto/watchlist\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:
127.0.0.1:56012 - \"GET /api/crypto/watchlist HTTP/1.1\" 200
OK\n{\"ts\":\"2025-10-12T21:19:26.663308+00:00\",\"level\":\"warning\",\"logger\":\"root\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood file not found:
data/market_mood.json\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"root\"}\n{\"ts\":\"2025-10-12T21:19:26.663401+00:00\",\"level\":\"info\",\"logger\":\"core.stage1_integration\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood stale,
updating...\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"core.stage1_integration\"}\n{\"ts\":\"2025-10-12T21:19:26.723632+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed
to get ticker 'SPY' reason: Expecting value: line 1 column 1 (char
0)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:26.784947+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"SPY:
No price data found, symbol may be delisted
(period=5d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:27.918520+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"QQQ:
No price data found, symbol may be delisted
(period=5d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:27.979336+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"Failed
to get ticker '^VIX' reason: Expecting value: line 1 column 1 (char
0)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:28.042510+00:00\",\"level\":\"error\",\"logger\":\"yfinance\",\"service\":\"ghost-wol\",\"msg\":\"^VIX:
No price data found, symbol may be delisted
(period=1d)\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"yfinance\"}\n{\"ts\":\"2025-10-12T21:19:28.043260+00:00\",\"level\":\"error\",\"logger\":\"root\",\"service\":\"ghost-wol\",\"msg\":\"Market
mood update failed: Insufficient SPY
data\",\"trace_id\":\"f3842f47-4025-4be7-9a9c-3c52959b2b10\",\"path\":\"/api/telegram/test\",\"method\":\"POST\",\"name\":\"root\"}\nINFO:
127.0.0.1:56026 - \"POST /api/telegram/test HTTP/1.1\" 200 OK\nINFO: 127.0.0.1:47566 - \"GET /cockpit HTTP/1.1\" 404 Not
Found\nINFO: 127.0.0.1:41194 - \"GET /cockpit HTTP/1.1\" 404 Not Found\nINFO: 127.0.0.1:45026 - \"GET /metrics
HTTP/1.1\" 200 OK\nINFO: Shutting down\nINFO: Waiting for application shutdown.\nINFO: Application shutdown
complete.\nINFO: Finished server process [169593]\n"}
{"ts": 1760305871, "file": "/tmp/ghost_final.log", "tail": "{\"ts\":\"2025-10-12T21:00:07.608882+00:00\",\"level\":\"error\",\"logger\":\"asyncio\",\"service\":\"ghost-wol\",\"msg\":\"Task exception was never retrieved\\nfuture: <Task finished name='Task-9' coro=<run_forever() done, defined at /workspaces/GHOST/core/workers/pattern_memory.py:100> exception=PermissionError(13, 'Permission denied')>\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"asyncio\",\"error_type\":\"PermissionError\",\"error\":\"[Errno 13] Permission denied: '/data'\"}\n{\"ts\":\"2025-10-12T21:00:07.608965+00:00\",\"level\":\"error\",\"logger\":\"asyncio\",\"service\":\"ghost-wol\",\"msg\":\"Task exception was never retrieved\\nfuture: <Task finished name='Task-10' coro=<run_forever() done, defined at /workspaces/GHOST/core/workers/reflex_trainer.py:89> exception=PermissionError(13, 'Permission denied')>\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"asyncio\",\"error_type\":\"PermissionError\",\"error\":\"[Errno 13] Permission denied: '/data'\"}\nINFO:     Application startup complete.\nINFO:     Uvicorn running on <<<<<http://0.0.0.0:5001>>>>> (Press CTRL+C to quit)\n{\"ts\":\"2025-10-12T21:00:08.770595+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"request\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"ghost\",\"component\":\"api\",\"status\":200,\"duration_ms\":1.59,\"client\":\"127.0.0.1\"}\n[GHOST INIT] ChatGPT Price Provider: DISABLED ('NoneType' object is not callable)\nINFO:     127.0.0.1:58208 - \"GET /health HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:00:18.821546+00:00\",\"level\":\"error\",\"logger\":\"core.edgar_integration\",\"service\":\"ghost-wol\",\"msg\":\"Error converting ticker WOLF to CIK: 404 Client Error: Not Found for url: <<<<<https://data.sec.gov/files/company_tickers.json\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"core.edgar_integration\"}\n{\"ts\":\"2025-10-12T21:00:18.821673+00:00\",\"level\":\"warning\",\"logger\":\"core.edgar_integration\",\"service\":\"ghost-wol\",\"msg\":\"Could>>>>> not find CIK for WOLF\",\"trace_id\":\"-\",\"path\":\"-\",\"method\":\"-\",\"name\":\"core.edgar_integration\"}\n[48h forecast] Generated: 42 at 1760302819\nINFO:     127.0.0.1:35480 - \"GET /health HTTP/1.1\" 200 OK\n{\"ts\":\"2025-10-12T21:03:21.690196+00:00\",\"level\":\"info\",\"logger\":\"ghost\",\"service\":\"ghost-wol\",\"msg\":\"Crypto providers initialized\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"ghost\"}\n{\"ts\":\"2025-10-12T21:03:21.972498+00:00\",\"level\":\"warning\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Binance fetch failed for BTC: 451 Client Error:  for url: <<<<<https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\n{\"ts\":\"2025-10-12T21:03:22.047659+00:00\",\"level\":\"info\",\"logger\":\"core.crypto.crypto_providers\",\"service\":\"ghost-wol\",\"msg\":\"Crypto>>>>> price quorum for BTC: $115023.26 (2 providers, 0.00% spread, 85% confidence)\",\"trace_id\":\"754ebde4-f5dd-4c31-ad89-e7b7a3f2bedf\",\"path\":\"/api/crypto/price/BTC\",\"method\":\"GET\",\"name\":\"core.crypto.crypto_providers\"}\nINFO:     127.0.0.1:35490 - \"GET /api/crypto/price/BTC HTTP/1.1\" 200 OK\nINFO:     127.0.0.1:37760 - \"GET /metrics HTTP/1.1\" 200 OK\nINFO:     Shutting down\nINFO:     Waiting for application shutdown.\nINFO:     Application shutdown complete.\nINFO:     Finished server process [161970]\n"}

```text

## UI Snapshot

- ui.html present: False


## News (3 items)

- 2025-10-04T08:31:00Z — Should You Buy Wolfspeed Stock Right Now? —


  <<<<<https://www.fool.com/investing/2025/10/04/should-you-buy-wolfspeed-right-now/?source=iedfolrf0000001>>>>>

- 2025-10-01T18:26:58Z — Why Is Wolfspeed Stock Plummeting Today? —


  <<<<<https://www.fool.com/investing/2025/10/01/why-is-wolfspeed-stock-plummeting-today/?source=iedfolrf0000001>>>>>

- 2025-09-30T12:20:04Z — Stock Market Today: Nasdaq, Dow Futures Slip As Shutdown


  Standoff Drags On—Cigna, Wolfspeed, Nike In Focus (UPDATED) —
  <<<<<https://www.benzinga.com/markets/equities/25/09/47936151/stock-market-today-sp-500-dow-futures-tumble-as-shutdown-standoff-drags-on-cigna-wolfspeed-nike->>>>>

## Telegram Echo Test

- Status: PENDING (provide TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID to capture message id)

#!/usr/bin/env bash
# Start Ghost WOLF server locally on port 8444 and verify readiness
set -euo pipefail
PORT=${PORT:-8444}
LOG=/tmp/ghost_local_${PORT}.log

# Kill existing server on port if any
if lsof -iTCP:${PORT} -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  pid=$(lsof -ti tcp:${PORT})
  echo "🔌 Port ${PORT} in use by PID ${pid}, killing..."
  kill ${pid} || true
  sleep 1
fi

# Launch
export CRYPTO_ENABLED=${CRYPTO_ENABLED:-1}
echo "🚀 Starting uvicorn wolf_app:APP on PORT=${PORT} (CRYPTO_ENABLED=${CRYPTO_ENABLED})..."
( PORT=${PORT} CRYPTO_ENABLED=${CRYPTO_ENABLED} /usr/local/bin/python3 -m uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT} ) >"${LOG}" 2>&1 &
PID=$!
echo ${PID} > /tmp/ghost_local_${PORT}.pid

# Wait for health
for i in {1..40}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PORT}/health" || true)
  if [ "${code}" = "200" ]; then
    echo "✅ Local server ready on http://127.0.0.1:${PORT} (PID ${PID})"
    echo "   Logs: ${LOG}"
    exit 0
  fi
  sleep 0.25
  [ $((i%10)) -eq 0 ] && echo "⏳ waiting for server (attempt ${i})..."
done

echo "❌ Server did not become healthy in time; showing last 40 log lines:"
 tail -n 40 "${LOG}" || true
exit 1

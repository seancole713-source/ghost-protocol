#!/usr/bin/env bash
# Stop Ghost WOLF server started on a given port (default 8444)
set -euo pipefail
PORT=${PORT:-8444}
PIDFILE=/tmp/ghost_local_${PORT}.pid

if [ -f "$PIDFILE" ]; then
  pid=$(cat "$PIDFILE")
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "🛑 Stopping PID ${pid} (PORT ${PORT})"
    kill "$pid" || true
    sleep 0.5
  fi
  rm -f "$PIDFILE"
fi

# Kill anything still listening on the port
if lsof -iTCP:${PORT} -sTCP:LISTEN -n -P >/dev/null 2>&1; then
  pid=$(lsof -ti tcp:${PORT})
  echo "🪓 Force-killing PID ${pid} on port ${PORT}"
  kill -9 ${pid} || true
fi

echo "✅ Stopped local server on port ${PORT}"

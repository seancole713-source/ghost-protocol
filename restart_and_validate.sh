#!/bin/bash
# Ghost Cockpit Restart & Validation Script
# Run after code changes to reload server and verify 90%+ ops

echo "=========================================="
echo "Ghost Cockpit Phase Upgrade → 90% Ops"
echo "Restart & Validation Script"
echo "=========================================="
echo

# Step 1: Show commit
echo "[1/5] Git Status:"
git log --oneline -1
echo

# Step 2: Compile check
echo "[2/5] Compiling wolf_app.py..."
python3 -m py_compile wolf_app.py
if [ $? -eq 0 ]; then
    echo "✅ Compilation successful"
else
    echo "❌ Compilation failed"
    exit 1
fi
echo

# Step 3: Kill old process
echo "[3/5] Stopping old server (if running)..."
OLD_PID=$(ps aux | grep 'python3 wolf_app.py' | grep -v grep | awk '{print $2}' | head -1)
if [ -n "$OLD_PID" ]; then
    echo "Found PID $OLD_PID, sending SIGTERM..."
    kill $OLD_PID
    sleep 3
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "Process still running, sending SIGKILL..."
        kill -9 $OLD_PID
    fi
    echo "✅ Old server stopped"
else
    echo "⚠️  No running server found (expected if PID 1 in Docker)"
fi
echo

# Step 4: Start new server (if not Docker PID 1)
echo "[4/5] Starting server..."
if [ "$$" == "1" ]; then
    echo "⚠️  Cannot start new server - running as PID 1 (Docker main process)"
    echo "    Need to restart container to load new code"
    echo "    Or manually exec: python3 wolf_app.py &"
else
    nohup python3 wolf_app.py > wolf_app.log 2>&1 &
    NEW_PID=$!
    echo "✅ Server started with PID $NEW_PID"
    echo "    Waiting 5s for startup..."
    sleep 5
fi
echo

# Step 5: Run endpoint tests
echo "[5/5] Testing endpoints..."
bash /app/test_endpoints.sh

echo
echo "=========================================="
echo "Restart Complete"
echo "=========================================="
echo
echo "Next steps:"
echo "1. Check logs: tail -f wolf_app.log"
echo "2. Monitor ops: curl http://127.0.0.1:8444/api/status"
echo "3. Verify AAPL: curl 'http://127.0.0.1:8444/api/price/diagnostics?symbol=AAPL'"
echo "4. Watch SSE: curl -sN http://127.0.0.1:8444/api/cockpit/stream | head -40"
echo

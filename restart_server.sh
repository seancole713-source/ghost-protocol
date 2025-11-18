#!/usr/bin/env bash
###############################################################################
# Restart Ghost Cockpit server and run comprehensive validation
# Handles PID 1 constraint in Docker containers
###############################################################################

set -euo pipefail

COLOR_RED='\033[0;31m'
COLOR_GREEN='\033[0;32m'
COLOR_YELLOW='\033[1;33m'
COLOR_CYAN='\033[0;36m'
COLOR_RESET='\033[0m'

echo -e "${COLOR_CYAN}╔════════════════════════════════════════════════════════════╗${COLOR_RESET}"
echo -e "${COLOR_CYAN}║ Ghost Cockpit Server Restart & Validation                 ║${COLOR_RESET}"
echo -e "${COLOR_CYAN}╚════════════════════════════════════════════════════════════╝${COLOR_RESET}"

# Check if running as PID 1 (Docker main process)
CURRENT_PID=$(ps -p 1 -o comm=)
if [[ "$CURRENT_PID" == *"python"* ]] || [[ "$CURRENT_PID" == *"uvicorn"* ]]; then
    echo -e "${COLOR_YELLOW}⚠️  Detected server running as PID 1 (Docker main process)${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}   Cannot hot-reload. Need container restart.${COLOR_RESET}"
    echo ""
    echo -e "${COLOR_CYAN}Options:${COLOR_RESET}"
    echo -e "  1) ${COLOR_GREEN}Railway:${COLOR_RESET} Trigger new deployment"
    echo -e "  2) ${COLOR_GREEN}Local Docker:${COLOR_RESET} docker restart <container_id>"
    echo -e "  3) ${COLOR_GREEN}Manual:${COLOR_RESET} Exit this script and restart server manually"
    echo ""
    read -p "Enter choice (1-3): " choice
    
    case $choice in
        1)
            echo -e "${COLOR_CYAN}→ Triggering Railway deployment...${COLOR_RESET}"
            if command -v railway &> /dev/null; then
                railway up --detach
                echo -e "${COLOR_GREEN}✓ Deployment triggered${COLOR_RESET}"
                echo -e "${COLOR_YELLOW}  Wait 60-90s for container restart${COLOR_RESET}"
                sleep 90
            else
                echo -e "${COLOR_RED}✗ Railway CLI not found${COLOR_RESET}"
                echo -e "  Install: npm i -g @railway/cli"
                exit 1
            fi
            ;;
        2)
            echo -e "${COLOR_CYAN}→ Finding Docker container...${COLOR_RESET}"
            CONTAINER_ID=$(docker ps --filter "ancestor=ghost-cockpit" --format "{{.ID}}" | head -1)
            if [[ -z "$CONTAINER_ID" ]]; then
                echo -e "${COLOR_RED}✗ Container not found${COLOR_RESET}"
                exit 1
            fi
            echo -e "${COLOR_CYAN}→ Restarting container $CONTAINER_ID...${COLOR_RESET}"
            docker restart "$CONTAINER_ID"
            echo -e "${COLOR_GREEN}✓ Container restarted${COLOR_RESET}"
            sleep 10
            ;;
        3)
            echo -e "${COLOR_YELLOW}⚠️  Manual restart required${COLOR_RESET}"
            echo -e "   After restart, run: bash /app/generate_ops_report.py"
            exit 0
            ;;
        *)
            echo -e "${COLOR_RED}✗ Invalid choice${COLOR_RESET}"
            exit 1
            ;;
    esac
else
    # Not PID 1, can restart normally
    echo -e "${COLOR_CYAN}→ Stopping existing server...${COLOR_RESET}"
    pkill -f "uvicorn wolf_app:APP" || true
    sleep 2
    
    echo -e "${COLOR_CYAN}→ Compiling wolf_app.py...${COLOR_RESET}"
    cd /app
    python3 -m py_compile wolf_app.py
    echo -e "${COLOR_GREEN}✓ Compilation successful${COLOR_RESET}"
    
    echo -e "${COLOR_CYAN}→ Starting server...${COLOR_RESET}"
    nohup python3 -m uvicorn wolf_app:APP --host 0.0.0.0 --port ${PORT:-8080} > server.log 2>&1 &
    SERVER_PID=$!
    echo -e "${COLOR_GREEN}✓ Server started (PID: $SERVER_PID)${COLOR_RESET}"
    sleep 5
fi

# Wait for server to be ready
echo ""
echo -e "${COLOR_CYAN}→ Waiting for server readiness...${COLOR_RESET}"
MAX_WAIT=30
WAITED=0
while ! curl -s http://127.0.0.1:${PORT:-8080}/api/status > /dev/null; do
    sleep 1
    WAITED=$((WAITED + 1))
    if [[ $WAITED -ge $MAX_WAIT ]]; then
        echo -e "${COLOR_RED}✗ Server not responding after ${MAX_WAIT}s${COLOR_RESET}"
        exit 1
    fi
done
echo -e "${COLOR_GREEN}✓ Server responding${COLOR_RESET}"

# Run acceptance tests
echo ""
echo -e "${COLOR_CYAN}→ Running acceptance tests...${COLOR_RESET}"
if [[ -f /app/acceptance_tests.sh ]]; then
    bash /app/acceptance_tests.sh
    TEST_EXIT=$?
else
    echo -e "${COLOR_YELLOW}⚠️  acceptance_tests.sh not found${COLOR_RESET}"
    TEST_EXIT=1
fi

# Generate ops report
echo ""
echo -e "${COLOR_CYAN}→ Generating OPS_REPORT.json...${COLOR_RESET}"
if [[ -f /app/generate_ops_report.py ]]; then
    python3 /app/generate_ops_report.py
    REPORT_EXIT=$?
else
    echo -e "${COLOR_YELLOW}⚠️  generate_ops_report.py not found${COLOR_RESET}"
    REPORT_EXIT=1
fi

# Summary
echo ""
echo -e "${COLOR_CYAN}╔════════════════════════════════════════════════════════════╗${COLOR_RESET}"
echo -e "${COLOR_CYAN}║ Validation Complete                                       ║${COLOR_RESET}"
echo -e "${COLOR_CYAN}╚════════════════════════════════════════════════════════════╝${COLOR_RESET}"

if [[ $TEST_EXIT -eq 0 ]] && [[ $REPORT_EXIT -eq 0 ]]; then
    echo -e "${COLOR_GREEN}✓ All tests passed - 100% operational${COLOR_RESET}"
    echo -e "${COLOR_GREEN}✓ OPS_REPORT.json generated${COLOR_RESET}"
    echo ""
    echo -e "${COLOR_CYAN}Next Steps:${COLOR_RESET}"
    echo -e "  1) Review: cat /app/OPS_REPORT.json | jq ."
    echo -e "  2) Deploy: git push railway main"
    echo -e "  3) Monitor: railway logs --follow"
    exit 0
else
    echo -e "${COLOR_RED}✗ Some tests failed${COLOR_RESET}"
    echo -e "${COLOR_YELLOW}  Review OPS_REPORT.json for details${COLOR_RESET}"
    exit 1
fi

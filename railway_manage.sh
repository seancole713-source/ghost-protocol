#!/bin/bash
# 🎛️ Railway Management Script
# Usage: ./railway_manage.sh [command]
#
# Commands:
#   deploy    - Deploy Ghost to Railway
#   logs      - View live logs
#   status    - Check deployment status
#   url       - Get deployment URL
#   health    - Test health endpoint
#   vars      - List environment variables
#   restart   - Restart the service
#   open      - Open in browser
#   restore   - Restore position data

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_header() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

cmd_deploy() {
    print_header "🚀 Deploying to Railway"
    railway up --detach
    echo -e "${GREEN}✓ Deployment triggered${NC}"
    echo ""
    echo "Monitor with: ./railway_manage.sh logs"
}

cmd_logs() {
    print_header "📋 Live Logs"
    railway logs
}

cmd_status() {
    print_header "📊 Deployment Status"
    railway status
}

cmd_url() {
    print_header "🌐 Deployment URL"
    DEPLOY_URL=$(railway domain 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)
    if [ -z "$DEPLOY_URL" ]; then
        echo -e "${RED}Could not retrieve URL${NC}"
        echo "Try: railway domain"
    else
        echo -e "${GREEN}$DEPLOY_URL${NC}"
        echo ""
        echo "Endpoints:"
        echo "  UI:           $DEPLOY_URL/"
        echo "  Health:       $DEPLOY_URL/health"
        echo "  Cockpit:      $DEPLOY_URL/api/cockpit"
        echo "  AI Memory:    $DEPLOY_URL/ai/memory/stats"
    fi
}

cmd_health() {
    print_header "🏥 Health Check"
    DEPLOY_URL=$(railway domain 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)
    
    if [ -z "$DEPLOY_URL" ]; then
        echo -e "${RED}Could not retrieve URL${NC}"
        exit 1
    fi
    
    echo "Testing: $DEPLOY_URL/health"
    echo ""
    
    RESPONSE=$(curl -s "$DEPLOY_URL/health" 2>&1)
    
    if [[ "$RESPONSE" == *"ok"* ]]; then
        echo -e "${GREEN}✓ Health check PASSED${NC}"
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
    else
        echo -e "${RED}✗ Health check FAILED${NC}"
        echo "$RESPONSE"
    fi
}

cmd_vars() {
    print_header "🔐 Environment Variables"
    railway variables
}

cmd_restart() {
    print_header "🔄 Restarting Service"
    railway restart
    echo -e "${GREEN}✓ Service restarted${NC}"
}

cmd_open() {
    print_header "🌐 Opening in Browser"
    railway open
}

cmd_restore() {
    print_header "💾 Restore Position Data"
    DEPLOY_URL=$(railway domain 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)
    
    if [ -z "$DEPLOY_URL" ]; then
        echo -e "${RED}Could not retrieve URL${NC}"
        exit 1
    fi
    
    echo "Restoring WOLF position: 8.41959051 @ $359.28"
    echo ""
    
    RESPONSE=$(curl -s -X POST "$DEPLOY_URL/api/position" \
        -H "Authorization: Bearer e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0" \
        -H "Content-Type: application/json" \
        -d '{"qty": 8.41959051, "avg_cost": 359.28}')
    
    if [[ "$RESPONSE" == *"ok"* ]] || [[ "$RESPONSE" == *"success"* ]]; then
        echo -e "${GREEN}✓ Position restored${NC}"
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
    else
        echo -e "${YELLOW}⚠ Response:${NC}"
        echo "$RESPONSE"
    fi
}

show_help() {
    cat << EOF
🎛️  Ghost Railway Manager

USAGE:
    ./railway_manage.sh [command]

COMMANDS:
    deploy      Deploy Ghost to Railway
    logs        View live deployment logs
    status      Check deployment status
    url         Get deployment URL and endpoints
    health      Test health endpoint
    vars        List environment variables
    restart     Restart the service
    open        Open deployment in browser
    restore     Restore position data (WOLF)

EXAMPLES:
    ./railway_manage.sh deploy        # Deploy now
    ./railway_manage.sh logs          # Watch logs
    ./railway_manage.sh health        # Test if alive
    ./railway_manage.sh restore       # Restore position

QUICK DEPLOY:
    ./deploy_ghost.sh                 # One-command setup + deploy
    ./redeploy.sh "Updated UI"        # Quick git push + deploy

EOF
}

# Main command router
case "${1:-help}" in
    deploy)
        cmd_deploy
        ;;
    logs)
        cmd_logs
        ;;
    status)
        cmd_status
        ;;
    url)
        cmd_url
        ;;
    health)
        cmd_health
        ;;
    vars)
        cmd_vars
        ;;
    restart)
        cmd_restart
        ;;
    open)
        cmd_open
        ;;
    restore)
        cmd_restore
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac

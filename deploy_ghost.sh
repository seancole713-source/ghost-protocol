#!/bin/bash
# 🚀 One-Command Railway Deploy Script for Ghost
# Usage: ./deploy_ghost.sh

set -e  # Exit on any error

echo "🚀 GHOST RAILWAY DEPLOYMENT"
echo "==========================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Step 1: Check if Railway CLI is installed
print_status "Checking Railway CLI..."
if ! command -v railway &> /dev/null; then
    print_error "Railway CLI not found"
    print_status "Installing Railway CLI..."
    
    if command -v npm &> /dev/null; then
        sudo npm install -g @railway/cli
        print_success "Railway CLI installed"
    else
        print_error "npm not found. Installing Node.js first..."
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        sudo apt-get install -y nodejs
        sudo npm install -g @railway/cli
        print_success "Node.js and Railway CLI installed"
    fi
else
    print_success "Railway CLI found ($(railway --version))"
fi

# Step 2: Check authentication
print_status "Checking Railway authentication..."
if ! railway whoami &> /dev/null; then
    print_warning "Not logged in to Railway"
    print_status "Opening browser for authentication..."
    railway login
    
    # Verify login
    if railway whoami &> /dev/null; then
        print_success "Logged in as: $(railway whoami)"
    else
        print_error "Login failed"
        exit 1
    fi
else
    print_success "Already logged in as: $(railway whoami)"
fi

# Step 3: Check if project is linked
print_status "Checking Railway project link..."
if [ ! -d ".railway" ]; then
    print_warning "Project not linked to Railway"
    print_status "Linking to ghost-protocol project..."
    
    # Try to link existing project
    railway link
    
    if [ ! -d ".railway" ]; then
        print_error "Failed to link project"
        print_warning "Run 'railway init' manually or select your project"
        exit 1
    fi
    print_success "Project linked"
else
    print_success "Project already linked"
fi

# Step 4: Check environment variables
print_status "Checking environment variables..."
VARS_COUNT=$(railway variables 2>/dev/null | grep -c "=" || echo "0")

if [ "$VARS_COUNT" -lt 5 ]; then
    print_warning "Environment variables not set or incomplete"
    print_status "You need to set environment variables manually:"
    echo ""
    echo "Run these commands (replace with your actual values):"
    echo ""
    echo "  railway variables set GHOST_API_TOKEN=your_token_here"
    echo "  railway variables set POLYGON_API_KEY=your_polygon_key"
    echo "  railway variables set ALPHAVANTAGE_API_KEY=your_alphavantage_key"
    echo "  railway variables set TELEGRAM_BOT_TOKEN=your_telegram_token"
    echo "  railway variables set TELEGRAM_CHAT_ID=your_chat_id"
    echo "  railway variables set GHOST_FOCUS_TICKER=WOLF"
    echo "  railway variables set WOLF_PERSIST_MODE=sqlite"
    echo "  railway variables set SIM_MODE=0"
    echo ""
    read -p "Press Enter after setting variables, or Ctrl+C to exit..."
    
    print_success "Environment variables should be set"
else
    print_success "Environment variables already configured"
fi

# Step 5: Deploy to Railway
print_status "Deploying to Railway..."
echo ""
railway up --detach

# Step 6: Wait for deployment
print_status "Waiting for deployment to complete..."
sleep 5

# Step 7: Get deployment URL
print_status "Getting deployment URL..."
DEPLOY_URL=$(railway domain 2>&1 | grep -o 'https://[^[:space:]]*' | head -1)

if [ -z "$DEPLOY_URL" ]; then
    print_warning "Could not retrieve URL automatically"
    print_status "Get your URL with: railway domain"
else
    print_success "Deployment URL: $DEPLOY_URL"
fi

# Step 8: Test deployment
if [ ! -z "$DEPLOY_URL" ]; then
    print_status "Testing health endpoint..."
    sleep 10  # Wait for service to start
    
    HEALTH_CHECK=$(curl -s "$DEPLOY_URL/health" 2>&1 || echo "failed")
    
    if [[ "$HEALTH_CHECK" == *"ok"* ]]; then
        print_success "Health check passed!"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo -e "${GREEN}✓ DEPLOYMENT SUCCESSFUL!${NC}"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "🌐 Ghost UI:        $DEPLOY_URL"
        echo "🏥 Health:          $DEPLOY_URL/health"
        echo "📊 Cockpit API:     $DEPLOY_URL/api/cockpit"
        echo "🤖 AI Memory:       $DEPLOY_URL/ai/memory/stats"
        echo ""
        echo "📋 Next Steps:"
        echo "   1. Visit $DEPLOY_URL in your browser"
        echo "   2. Restore position data (see below)"
        echo "   3. Monitor logs: railway logs"
        echo ""
        echo "💾 Restore Position Data:"
        echo "   curl -X POST '$DEPLOY_URL/api/position' \\"
        echo "     -H 'Authorization: Bearer e3c4a2f7-91d9-44e8-b7a2-f61c09f8d9d0' \\"
        echo "     -H 'Content-Type: application/json' \\"
        echo "     -d '{\"qty\": 8.41959051, \"avg_cost\": 359.28}'"
        echo ""
    else
        print_warning "Health check pending (service may still be starting)"
        print_status "Check status with: railway logs"
        echo ""
        echo "Your deployment is in progress. Check logs:"
        echo "  railway logs"
        echo ""
        echo "Get your URL:"
        echo "  railway domain"
    fi
fi

# Step 9: Show useful commands
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📚 USEFUL COMMANDS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  railway logs              # View live logs"
echo "  railway status            # Check deployment status"
echo "  railway domain            # Get your app URL"
echo "  railway open              # Open app in browser"
echo "  railway variables         # List environment variables"
echo "  ./deploy_ghost.sh         # Redeploy (this script)"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

print_success "Deployment script completed!"

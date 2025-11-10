#!/bin/bash
# 🔄 Quick Redeploy Script - Use after initial setup
# Usage: ./redeploy.sh [optional commit message]

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔄 Quick Ghost Redeploy${NC}"
echo "======================="
echo ""

# Get commit message or use default
COMMIT_MSG="${1:-Quick update}"

# Step 1: Stage all changes
echo -e "${BLUE}▶${NC} Staging changes..."
git add -A

# Step 2: Check if there are changes
if git diff --staged --quiet; then
    echo -e "${GREEN}✓${NC} No changes to commit"
else
    # Step 3: Commit
    echo -e "${BLUE}▶${NC} Committing: $COMMIT_MSG"
    git commit -m "$COMMIT_MSG"
    
    # Step 4: Push
    echo -e "${BLUE}▶${NC} Pushing to GitHub..."
    git push origin main
    
    echo -e "${GREEN}✓${NC} Pushed to GitHub"
fi

# Step 5: Deploy to Railway
echo -e "${BLUE}▶${NC} Deploying to Railway..."
railway up --detach

echo ""
echo -e "${GREEN}✓ Deployment triggered!${NC}"
echo ""
echo "Monitor progress:"
echo "  railway logs"
echo ""
echo "Check status:"
echo "  railway status"
echo ""
echo "Get URL:"
echo "  railway domain"

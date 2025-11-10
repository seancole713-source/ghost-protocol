#!/bin/bash

# Ghost Trading System Restore Script
# Restores from backup

BACKUP_DIR="${BACKUP_DIR:-/opt/backups/ghost}"
GHOST_DIR="${GHOST_DIR:-/opt/GHOST}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# List available backups
log "📋 Available backups:"
ls -lh "$BACKUP_DIR"/ghost_backup_*.tar.gz | tail -10

echo
read -p "Enter backup filename to restore (or press Enter for latest): " BACKUP_FILE

if [ -z "$BACKUP_FILE" ]; then
    BACKUP_FILE=$(ls -t "$BACKUP_DIR"/ghost_backup_*.tar.gz | head -1)
    log "Using latest backup: $(basename $BACKUP_FILE)"
fi

if [ ! -f "$BACKUP_FILE" ]; then
    BACKUP_FILE="$BACKUP_DIR/$BACKUP_FILE"
fi

if [ ! -f "$BACKUP_FILE" ]; then
    log "❌ Backup file not found: $BACKUP_FILE"
    exit 1
fi

# Confirm
echo
log "⚠️  This will restore Ghost to the backup: $(basename $BACKUP_FILE)"
read -p "Continue? (yes/no): " CONFIRM

if [ "$CONFIRM" != "yes" ]; then
    log "Restore cancelled"
    exit 0
fi

# Stop Ghost
log "🛑 Stopping Ghost..."
if command -v docker-compose &> /dev/null; then
    cd "$GHOST_DIR"
    docker-compose down
elif command -v systemctl &> /dev/null; then
    systemctl stop ghost
else
    pkill -f "uvicorn.*wolf_app"
fi

# Backup current state
log "💾 Backing up current state..."
CURRENT_BACKUP="$BACKUP_DIR/pre_restore_$(date +%Y%m%d_%H%M%S).tar.gz"
cd "$GHOST_DIR/data"
tar -czf "$CURRENT_BACKUP" *.db 2>/dev/null
log "✅ Current state saved to: $(basename $CURRENT_BACKUP)"

# Extract backup
log "📦 Extracting backup..."
TEMP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"

# Restore databases
log "🔄 Restoring databases..."
for db in wolf ai_memory ghost_ai; do
    DB_FILE=$(ls -t "$TEMP_DIR"/${db}_*.db 2>/dev/null | head -1)
    if [ -f "$DB_FILE" ]; then
        cp "$DB_FILE" "$GHOST_DIR/data/${db}.db"
        log "✅ Restored ${db}.db"
    fi
done

# Restore environment (optional)
ENV_FILE=$(ls -t "$TEMP_DIR"/env_*.txt 2>/dev/null | head -1)
if [ -f "$ENV_FILE" ]; then
    read -p "Restore environment configuration? (y/n): " RESTORE_ENV
    if [ "$RESTORE_ENV" = "y" ]; then
        cp "$ENV_FILE" "$GHOST_DIR/.env.production"
        log "✅ Restored environment configuration"
    fi
fi

# Cleanup
rm -rf "$TEMP_DIR"

# Start Ghost
log "🚀 Starting Ghost..."
if command -v docker-compose &> /dev/null; then
    cd "$GHOST_DIR"
    docker-compose up -d
elif command -v systemctl &> /dev/null; then
    systemctl start ghost
else
    cd "$GHOST_DIR"
    nohup uvicorn wolf_app:app --host 0.0.0.0 --port 5000 > /tmp/ghost.log 2>&1 &
fi

# Wait and verify
sleep 10
log "🏥 Verifying health..."
if curl -f -s http://localhost:5000/health > /dev/null 2>&1; then
    log "✅ Ghost restored successfully!"
else
    log "⚠️  Health check failed. Check logs for details."
fi

log "✅ Restore complete!"
echo
log "📊 To verify restoration:"
log "  curl http://localhost:5000/health/detailed | jq '.'"

#!/bin/bash

# Ghost Trading System Backup Script
# Backs up databases and configuration

BACKUP_DIR="${BACKUP_DIR:-/opt/backups/ghost}"
GHOST_DIR="${GHOST_DIR:-/opt/GHOST}"
RETENTION_DAYS="${RETENTION_DAYS:-7}"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Backup databases
log "📦 Starting Ghost backup..."

if [ -f "$GHOST_DIR/data/wolf.db" ]; then
    cp "$GHOST_DIR/data/wolf.db" "$BACKUP_DIR/wolf_${DATE}.db"
    log "✅ Backed up wolf.db"
fi

if [ -f "$GHOST_DIR/data/ai_memory.db" ]; then
    cp "$GHOST_DIR/data/ai_memory.db" "$BACKUP_DIR/ai_memory_${DATE}.db"
    log "✅ Backed up ai_memory.db"
fi

if [ -f "$GHOST_DIR/data/ghost_ai.db" ]; then
    cp "$GHOST_DIR/data/ghost_ai.db" "$BACKUP_DIR/ghost_ai_${DATE}.db"
    log "✅ Backed up ghost_ai.db"
fi

# Backup environment file (without sensitive data exposed in filename)
if [ -f "$GHOST_DIR/.env.production" ]; then
    cp "$GHOST_DIR/.env.production" "$BACKUP_DIR/env_${DATE}.txt"
    log "✅ Backed up environment configuration"
fi

# Create compressed archive
cd "$BACKUP_DIR"
tar -czf "ghost_backup_${DATE}.tar.gz" \
    wolf_${DATE}.db \
    ai_memory_${DATE}.db \
    ghost_ai_${DATE}.db \
    env_${DATE}.txt 2>/dev/null

# Remove individual files after archiving
rm -f wolf_${DATE}.db ai_memory_${DATE}.db ghost_ai_${DATE}.db env_${DATE}.txt

log "✅ Created compressed backup: ghost_backup_${DATE}.tar.gz"

# Calculate backup size
SIZE=$(du -h "ghost_backup_${DATE}.tar.gz" | cut -f1)
log "📊 Backup size: $SIZE"

# Remove old backups
log "🧹 Cleaning up old backups (keeping last $RETENTION_DAYS days)..."
find "$BACKUP_DIR" -name "ghost_backup_*.tar.gz" -mtime +$RETENTION_DAYS -delete

REMAINING=$(ls -1 "$BACKUP_DIR"/ghost_backup_*.tar.gz 2>/dev/null | wc -l)
log "📁 Total backups: $REMAINING"

# Optional: Upload to cloud storage
if [ -n "$S3_BUCKET" ]; then
    log "☁️  Uploading to S3..."
    aws s3 cp "ghost_backup_${DATE}.tar.gz" "s3://$S3_BUCKET/ghost-backups/"
    log "✅ Uploaded to S3"
fi

log "✅ Backup complete!"

#!/usr/bin/env python3
"""
Railway Daily Backup Script
============================
Backs up Ghost databases to Railway persistent volume.
Run as cron job: 0 3 * * * (3 AM UTC daily)
"""

import json
import os
import shutil
import time
from pathlib import Path

# Configuration
BACKUP_DIR = Path("/app/backups")
MAX_BACKUPS = 7  # Keep last 7 days
DATABASES = ["wolf.db", "ai_memory.db", "goals_log.db", "data/wolf.db", "data/ghost_ai.db"]


def backup_databases():
    """Backup all Ghost databases with timestamp."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_count = 0
    errors = []

    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)

    print(f"🔄 Starting backup at {timestamp}")

    # Backup each database
    for db_path in DATABASES:
        if os.path.exists(db_path):
            try:
                db_name = os.path.basename(db_path)
                backup_path = BACKUP_DIR / f"{db_name}.{timestamp}.bak"

                # Copy database
                shutil.copy2(db_path, backup_path)

                # Get file size
                size_mb = os.path.getsize(backup_path) / (1024 * 1024)

                print(f"✅ Backed up {db_path} → {backup_path.name} ({size_mb:.2f} MB)")
                backup_count += 1

            except Exception as e:
                error_msg = f"❌ Failed to backup {db_path}: {e}"
                print(error_msg)
                errors.append(error_msg)
        else:
            print(f"⚠️  Skipping {db_path} (not found)")

    # Create backup manifest
    manifest = {
        "timestamp": timestamp,
        "backup_count": backup_count,
        "errors": errors,
        "databases": DATABASES,
    }

    manifest_path = BACKUP_DIR / f"manifest.{timestamp}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"📋 Backup manifest: {manifest_path.name}")

    return backup_count, errors


def cleanup_old_backups():
    """Remove backups older than MAX_BACKUPS days."""
    if not BACKUP_DIR.exists():
        return 0

    print(f"\n🧹 Cleaning up backups (keeping last {MAX_BACKUPS} days)")

    # Get all backup files sorted by modification time
    backup_files = sorted(BACKUP_DIR.glob("*.bak"), key=lambda p: p.stat().st_mtime, reverse=True)

    # Keep track of backups per database
    db_backups = {}
    deleted_count = 0

    for backup_file in backup_files:
        # Extract database name (before timestamp)
        db_name = backup_file.stem.rsplit(".", 1)[0]

        if db_name not in db_backups:
            db_backups[db_name] = []

        db_backups[db_name].append(backup_file)

    # Delete old backups for each database
    for _db_name, backups in db_backups.items():
        if len(backups) > MAX_BACKUPS:
            for old_backup in backups[MAX_BACKUPS:]:
                try:
                    old_backup.unlink()
                    print(f"🗑️  Deleted old backup: {old_backup.name}")
                    deleted_count += 1
                except Exception as e:
                    print(f"❌ Failed to delete {old_backup.name}: {e}")

    # Cleanup old manifests
    manifest_files = sorted(
        BACKUP_DIR.glob("manifest.*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )

    if len(manifest_files) > MAX_BACKUPS:
        for old_manifest in manifest_files[MAX_BACKUPS:]:
            try:
                old_manifest.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"❌ Failed to delete {old_manifest.name}: {e}")

    return deleted_count


def get_backup_stats():
    """Get statistics about current backups."""
    if not BACKUP_DIR.exists():
        return None

    backup_files = list(BACKUP_DIR.glob("*.bak"))
    total_size = sum(f.stat().st_size for f in backup_files)

    stats = {
        "total_backups": len(backup_files),
        "total_size_mb": total_size / (1024 * 1024),
        "backup_dir": str(BACKUP_DIR),
    }

    return stats


def main():
    """Main backup routine."""
    print("=" * 60)
    print("Ghost Railway Daily Backup")
    print("=" * 60)

    # Show current backup stats
    stats = get_backup_stats()
    if stats:
        print("\n📊 Current backup stats:")
        print(f"   Total backups: {stats['total_backups']}")
        print(f"   Total size: {stats['total_size_mb']:.2f} MB")
        print(f"   Location: {stats['backup_dir']}\n")

    # Perform backup
    backup_count, errors = backup_databases()

    # Cleanup old backups
    deleted_count = cleanup_old_backups()

    # Summary
    print("\n" + "=" * 60)
    print("📦 Backup Summary")
    print("=" * 60)
    print(f"✅ Databases backed up: {backup_count}")
    print(f"❌ Errors: {len(errors)}")
    print(f"🗑️  Old backups deleted: {deleted_count}")

    # Show new stats
    stats = get_backup_stats()
    if stats:
        print(f"📊 New total backups: {stats['total_backups']}")
        print(f"💾 Total size: {stats['total_size_mb']:.2f} MB")

    print("=" * 60)

    # Exit with error code if backups failed
    if errors:
        exit(1)
    else:
        print("✅ Backup completed successfully!")
        exit(0)


if __name__ == "__main__":
    main()

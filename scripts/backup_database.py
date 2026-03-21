#!/usr/bin/env python3
"""
Ghost Protocol - Database Backup Script
Creates timestamped backups of PostgreSQL database
"""

import os
import subprocess
import datetime
from pathlib import Path

# Database connection from environment
DB_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@localhost:5432/ghost")
BACKUP_DIR = Path("/workspaces/ghost-protocol/backups")
BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# Parse connection URL
def parse_db_url(url):
    """Extract connection details from DATABASE_URL"""
    # Format: postgresql://user:pass@host:port/dbname
    url = url.replace("postgresql://", "")
    if "@" in url:
        auth, rest = url.split("@", 1)
        user, password = auth.split(":", 1) if ":" in auth else (auth, "")
        host_port, dbname = rest.split("/", 1)
        host, port = host_port.split(":", 1) if ":" in host_port else (host_port, "5432")
    else:
        user, password, host, port, dbname = "postgres", "", "localhost", "5432", url.split("/")[-1]
    
    return {
        "user": user,
        "password": password,
        "host": host,
        "port": port,
        "dbname": dbname
    }


def backup_database():
    """Create pg_dump backup with timestamp"""
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_file = BACKUP_DIR / f"ghost_backup_{timestamp}.sql"
    
    print(f"🔄 Starting database backup...")
    print(f"📁 Backup file: {backup_file}")
    
    db_config = parse_db_url(DB_URL)
    
    # Build pg_dump command
    cmd = [
        "pg_dump",
        "-h", db_config["host"],
        "-p", db_config["port"],
        "-U", db_config["user"],
        "-d", db_config["dbname"],
        "-F", "p",  # Plain text format
        "-f", str(backup_file),
        "--verbose"
    ]
    
    # Set password environment variable
    env = os.environ.copy()
    if db_config["password"]:
        env["PGPASSWORD"] = db_config["password"]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            size_mb = backup_file.stat().st_size / (1024 * 1024)
            print(f"✅ Backup successful: {backup_file.name} ({size_mb:.1f} MB)")
            
            # Clean old backups (keep last 7 days)
            clean_old_backups()
            
            return backup_file
        else:
            print(f"❌ Backup failed: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ Backup timed out after 5 minutes")
        return None
    except Exception as e:
        print(f"❌ Backup error: {e}")
        return None


def clean_old_backups(keep_days=7):
    """Remove backups older than N days"""
    cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=keep_days)
    deleted = 0
    
    for backup in BACKUP_DIR.glob("ghost_backup_*.sql"):
        # Parse timestamp from filename
        try:
            timestamp_str = backup.stem.split("_", 2)[2]  # ghost_backup_YYYYMMDD_HHMMSS
            file_time = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            if file_time < cutoff:
                backup.unlink()
                deleted += 1
        except (ValueError, IndexError):
            continue
    
    if deleted > 0:
        print(f"🗑️  Cleaned {deleted} old backup(s)")


def restore_database(backup_file):
    """Restore database from backup file"""
    print(f"🔄 Restoring from: {backup_file}")
    
    db_config = parse_db_url(DB_URL)
    
    cmd = [
        "psql",
        "-h", db_config["host"],
        "-p", db_config["port"],
        "-U", db_config["user"],
        "-d", db_config["dbname"],
        "-f", str(backup_file)
    ]
    
    env = os.environ.copy()
    if db_config["password"]:
        env["PGPASSWORD"] = db_config["password"]
    
    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Restore successful")
            return True
        else:
            print(f"❌ Restore failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Restore error: {e}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        if len(sys.argv) < 3:
            print("Usage: python backup_database.py restore <backup_file>")
            sys.exit(1)
        
        backup_path = Path(sys.argv[2])
        if not backup_path.exists():
            print(f"❌ Backup file not found: {backup_path}")
            sys.exit(1)
        
        restore_database(backup_path)
    else:
        # Default: create backup
        backup_database()

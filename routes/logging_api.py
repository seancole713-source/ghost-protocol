"""
Phase 5.3: Logging Dashboard API

Provides structured log viewing and filtering endpoints.
Displays recent logs with severity, module, timestamp filtering.
"""

from fastapi import APIRouter, Query
from typing import Optional
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

router = APIRouter()


def parse_log_file(
    log_path: str, 
    limit: int = 100, 
    level: Optional[str] = None,
    module: Optional[str] = None,
    since_minutes: Optional[int] = None
) -> list[dict]:
    """
    Parse log file and return structured entries.
    
    Args:
        log_path: Path to log file
        limit: Max entries to return
        level: Filter by log level (INFO, WARNING, ERROR, etc.)
        module: Filter by module name
        since_minutes: Only show logs from last N minutes
    
    Returns:
        List of log entries with timestamp, level, module, message
    """
    if not os.path.exists(log_path):
        return []
    
    entries = []
    cutoff_time = None
    if since_minutes:
        cutoff_time = datetime.now() - timedelta(minutes=since_minutes)
    
    # Log format: 2026-03-21 14:32:15 INFO     [core.auto_prediction_loop] Message here
    log_pattern = r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s+(\w+)\s+\[([^\]]+)\]\s+(.+)"
    
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            # Read last N lines efficiently (reverse read)
            lines = f.readlines()[-1000:]  # Last 1000 lines
            
            for line in reversed(lines):
                match = re.match(log_pattern, line.strip())
                if not match:
                    continue
                
                timestamp_str, log_level, log_module, message = match.groups()
                
                # Parse timestamp
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                except ValueError:
                    continue
                
                # Apply filters
                if level and log_level.upper() != level.upper():
                    continue
                if module and module.lower() not in log_module.lower():
                    continue
                if cutoff_time and timestamp < cutoff_time:
                    continue
                
                entries.append({
                    "timestamp": timestamp_str,
                    "level": log_level,
                    "module": log_module,
                    "message": message
                })
                
                if len(entries) >= limit:
                    break
    
    except Exception as e:
        entries.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "level": "ERROR",
            "module": "logging_api",
            "message": f"Failed to parse log file: {e}"
        })
    
    return entries


@router.get("/api/logs/recent")
async def get_recent_logs(
    limit: int = Query(100, ge=1, le=500, description="Number of log entries"),
    level: Optional[str] = Query(None, description="Filter by level (INFO, WARNING, ERROR)"),
    module: Optional[str] = Query(None, description="Filter by module name"),
    since: Optional[int] = Query(None, ge=1, description="Show logs from last N minutes")
):
    """
    Get recent log entries with optional filtering.
    
    Query Parameters:
    - limit: Max entries (1-500, default 100)
    - level: Filter by severity (INFO, WARNING, ERROR, CRITICAL)
    - module: Filter by module name (substring match)
    - since: Only show logs from last N minutes
    
    Returns:
    {
        "ok": true,
        "count": 42,
        "logs": [
            {
                "timestamp": "2026-03-21 14:32:15",
                "level": "INFO",
                "module": "core.auto_prediction_loop",
                "message": "Cycle completed"
            }
        ]
    }
    """
    # Determine log file location
    log_file = os.getenv("LOG_FILE", "ghost_protocol.log")
    if not os.path.isabs(log_file):
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), log_file)
    
    logs = parse_log_file(
        log_path=log_file,
        limit=limit,
        level=level,
        module=module,
        since_minutes=since
    )
    
    return {
        "ok": True,
        "count": len(logs),
        "filters": {
            "level": level,
            "module": module,
            "since_minutes": since
        },
        "logs": logs
    }


@router.get("/api/logs/errors")
async def get_error_logs(
    limit: int = Query(50, ge=1, le=200),
    since: Optional[int] = Query(60, description="Show errors from last N minutes")
):
    """
    Get recent ERROR and CRITICAL level logs.
    Quick endpoint for checking what's broken.
    """
    log_file = os.getenv("LOG_FILE", "ghost_protocol.log")
    if not os.path.isabs(log_file):
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), log_file)
    
    # Get errors first, then criticals
    errors = parse_log_file(log_file, limit=limit, level="ERROR", since_minutes=since)
    criticals = parse_log_file(log_file, limit=limit, level="CRITICAL", since_minutes=since)
    
    combined = criticals + errors
    combined.sort(key=lambda x: x["timestamp"], reverse=True)
    
    return {
        "ok": True,
        "count": len(combined),
        "errors": combined[:limit]
    }


@router.get("/api/logs/summary")
async def get_log_summary(since: int = Query(60, description="Last N minutes")):
    """
    Get log level distribution for the last N minutes.
    Useful for health dashboard.
    
    Returns:
    {
        "ok": true,
        "since_minutes": 60,
        "summary": {
            "INFO": 142,
            "WARNING": 5,
            "ERROR": 2,
            "CRITICAL": 0
        }
    }
    """
    log_file = os.getenv("LOG_FILE", "ghost_protocol.log")
    if not os.path.isabs(log_file):
        log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), log_file)
    
    logs = parse_log_file(log_file, limit=1000, since_minutes=since)
    
    summary = {
        "DEBUG": 0,
        "INFO": 0,
        "WARNING": 0,
        "ERROR": 0,
        "CRITICAL": 0
    }
    
    for log in logs:
        level = log["level"].upper()
        if level in summary:
            summary[level] += 1
    
    return {
        "ok": True,
        "since_minutes": since,
        "total_logs": len(logs),
        "summary": summary
    }

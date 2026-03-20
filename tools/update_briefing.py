#!/usr/bin/env python3
"""
tools/update_briefing.py — Append entries to PROJECT_STATE.py changelog
========================================================================
Usage:
    python tools/update_briefing.py "AgentName" "Summary of what was done"

Example:
    python tools/update_briefing.py "Claude-Browser" "Fixed Bug #23, created HANDOFF.md"

This script finds the CHANGE_LOG section in PROJECT_STATE.py and appends
a new timestamped entry. It also updates the Last Updated date at the top.
"""

import sys
import os
import re
from datetime import datetime, timezone


def find_project_state():
    """Locate PROJECT_STATE.py relative to this script."""
    # Script lives in tools/, PROJECT_STATE.py is at repo root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    path = os.path.join(repo_root, "PROJECT_STATE.py")
    if os.path.exists(path):
        return path
    # Fallback: check current working directory
    cwd_path = os.path.join(os.getcwd(), "PROJECT_STATE.py")
    if os.path.exists(cwd_path):
        return cwd_path
    return None


def update_changelog(agent_name: str, summary: str):
    """Append a changelog entry to PROJECT_STATE.py."""
    path = find_project_state()
    if not path:
        print("ERROR: PROJECT_STATE.py not found.")
        print("  Checked: tools/../PROJECT_STATE.py and ./PROJECT_STATE.py")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M UTC")
    
    # Build the new changelog entry
    new_entry = f'    "{date_str} [{agent_name}] ({time_str}): {summary}",'

    # Find the CHANGE_LOG list and insert after the opening bracket
    # Pattern: CHANGE_LOG = [  ... existing entries ... ]
    pattern = r'(CHANGE_LOG\s*=\s*\[)'
    match = re.search(pattern, content)
    if not match:
        print("ERROR: Could not find CHANGE_LOG = [ in PROJECT_STATE.py")
        sys.exit(1)

    # Insert new entry right after the opening bracket
    insert_pos = match.end()
    content = content[:insert_pos] + "\n" + new_entry + content[insert_pos:]

    # Update the "Last Updated" date in the docstring
    content = re.sub(
        r'Last Updated:\s*\d{4}-\d{2}-\d{2}',
        f'Last Updated: {date_str}',
        content,
        count=1
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"OK: Changelog updated in {path}")
    print(f"  Agent: {agent_name}")
    print(f"  Date:  {date_str} {time_str}")
    print(f"  Entry: {summary}")


def update_handoff(agent_name: str, handoff_text: str):
    """Update the LAST_SESSION_HANDOFF block in PROJECT_STATE.py."""
    path = find_project_state()
    if not path:
        print("ERROR: PROJECT_STATE.py not found.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y-%m-%d %H:%M UTC")

    # Replace the LAST_SESSION_HANDOFF value
    new_handoff = f'''LAST_SESSION_HANDOFF = """
Agent: {agent_name}
Date:  {date_str}

{handoff_text}
"""'''

    # Pattern to match the existing LAST_SESSION_HANDOFF block
    pattern = r'LAST_SESSION_HANDOFF\s*=\s*"""[\s\S]*?"""'
    if re.search(pattern, content):
        content = re.sub(pattern, new_handoff, content, count=1)
    else:
        print("WARNING: Could not find LAST_SESSION_HANDOFF block. Appending instead.")
        content += "\n\n" + new_handoff + "\n"

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"OK: Handoff updated for agent {agent_name}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python tools/update_briefing.py <agent_name> <summary>")
        print()
        print("Commands:")
        print("  python tools/update_briefing.py <agent> <summary>     # Update changelog")
        print("  python tools/update_briefing.py --handoff <agent> <text>  # Update handoff")
        sys.exit(1)

    if sys.argv[1] == "--handoff":
        if len(sys.argv) < 4:
            print("Usage: python tools/update_briefing.py --handoff <agent_name> <handoff_text>")
            sys.exit(1)
        update_handoff(sys.argv[2], sys.argv[3])
    else:
        agent_name = sys.argv[1]
        summary = sys.argv[2]
        update_changelog(agent_name, summary)

#!/usr/bin/env python3
"""Quick script to replace timezone.utc with UTC in ghost_agent_loop.py"""

filepath = "/workspaces/GHOST/ghost_agent_loop.py"

with open(filepath) as f:
    content = f.read()

original_count = content.count("timezone.utc")
content = content.replace("timezone.utc", "UTC")
new_count = content.count("timezone.utc")

with open(filepath, "w") as f:
    f.write(content)

print(f"✅ Replaced {original_count - new_count} instances of 'timezone.utc' with 'UTC'")
print(f"   Remaining 'timezone.utc' references: {new_count}")

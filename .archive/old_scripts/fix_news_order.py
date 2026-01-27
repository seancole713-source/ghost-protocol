#!/usr/bin/env python3
"""Fix news endpoint ordering - move helper function before endpoints that use it."""

# Read the file
with open("wolf_app.py") as f:
    lines = f.readlines()

# Find the helper function (around line 18391)
helper_start = None
helper_end = None
for i, line in enumerate(lines):
    if "async def _get_news_feed" in line and helper_start is None:
        helper_start = i
    if helper_start is not None and helper_end is None:
        if (
            line.strip()
            and not line.startswith(" ")
            and not line.startswith("\t")
            and i > helper_start
        ):
            helper_end = i
            break

# If we didn't find an end, look for the next function
if helper_start and not helper_end:
    for i in range(helper_start + 1, len(lines)):
        if (
            lines[i].startswith("async def ")
            or lines[i].startswith("def ")
            or lines[i].startswith("@APP.")
        ):
            helper_end = i
            break

# Extract helper function
if helper_start and helper_end:
    helper_lines = lines[helper_start:helper_end]
    print(f"Found helper function at lines {helper_start + 1}-{helper_end}")

    # Remove helper from its current location
    del lines[helper_start:helper_end]

    # Find where to insert it (before line 14589, which is now shifted)
    # Look for @APP.get("/api/news")
    insert_pos = None
    for i, line in enumerate(lines):
        if '@APP.get("/api/news")' in line and "async def api_news" in lines[i + 1]:
            insert_pos = i
            break

    if insert_pos:
        # Insert helper function before the endpoint
        lines[insert_pos:insert_pos] = ["\n"] + helper_lines + ["\n"]
        print(f"Inserted helper function before endpoint at line {insert_pos + 1}")

        # Write back
        with open("wolf_app.py", "w") as f:
            f.writelines(lines)

        print("✅ Fixed news endpoint ordering")
    else:
        print("❌ Could not find news endpoint")
else:
    print("❌ Could not find helper function")

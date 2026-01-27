#!/usr/bin/env python3
"""
Quick fix: Add API endpoint to serve simulation data
Run this to patch wolf_app.py with a /api/simulation_data endpoint
"""

import sys

# Read wolf_app.py
with open("wolf_app.py") as f:
    content = f.read()

# Check if endpoint already exists
if "/api/simulation_data" in content:
    print("✅ Simulation data endpoint already exists")
    sys.exit(0)

# Find a good place to add the endpoint (after imports, before routes)
insert_marker = '@APP.get("/api/status")'

if insert_marker not in content:
    print("❌ Could not find insertion point in wolf_app.py")
    sys.exit(1)

# New endpoint code
new_endpoint = '''
# Simulation data endpoint
@APP.get("/api/simulation_data")
async def api_simulation_data():
    """Serve simulation data for UI validation testing."""
    import json
    import os

    sim_file = os.path.join(os.path.dirname(__file__), 'public', 'simulation_data.json')

    if not os.path.exists(sim_file):
        return {"error": "Simulation data not found", "hint": "Run: python3 generate_simulation_data.py"}

    with open(sim_file, 'r') as f:
        data = json.load(f)

    return data

'''

# Insert before the status endpoint
content = content.replace(insert_marker, new_endpoint + insert_marker)

# Write back
with open("wolf_app.py", "w") as f:
    f.write(content)

print("✅ Added /api/simulation_data endpoint to wolf_app.py")
print("")
print("Next steps:")
print("  1. Server will auto-reload (--reload flag)")
print("  2. Test: curl http://localhost:5000/api/simulation_data | jq 'keys'")
print("  3. Use in JavaScript: fetch('/api/simulation_data').then(r => r.json())")
print("")

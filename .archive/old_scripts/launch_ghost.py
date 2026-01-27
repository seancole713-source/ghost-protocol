#!/usr/bin/env python3
"""
Ghost Server Launcher
Properly loads secrets and starts uvicorn
"""

import os

# Load secrets.env
secrets_file = "/workspaces/GHOST/secrets.env"
if os.path.exists(secrets_file):
    with open(secrets_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                # Remove quotes
                value = value.strip().strip('"').strip("'")
                os.environ[key] = value
                print(f"Loaded: {key}")

# Set Prometheus dir
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/ghost_prom"
os.makedirs("/tmp/ghost_prom", exist_ok=True)

# Verify keys
polygon = os.getenv("POLYGON_API_KEY", "")
alpha = os.getenv("ALPHAVANTAGE_API_KEY", "")
print(f"\nPOLYGON_API_KEY: {'SET' if polygon else 'MISSING'}")
print(f"ALPHAVANTAGE_API_KEY: {'SET' if alpha else 'MISSING'}")

if not polygon or not alpha:
    print("\n⚠️  WARNING: API keys not found in secrets.env!")
    print("Pull them from Railway → Variables and add to secrets.env or export them before launch.")
    print("Example:")
    print("POLYGON_API_KEY=\"$(railway variables get POLYGON_API_KEY)\"")
    print("ALPHAVANTAGE_API_KEY=\"$(railway variables get ALPHAVANTAGE_API_KEY)\"\n")

# Start uvicorn (without --reload to preserve environment)
print("\nStarting Ghost server...\n")
os.execvp("uvicorn", ["uvicorn", "wolf_app:app", "--host", "0.0.0.0", "--port", "5000"])

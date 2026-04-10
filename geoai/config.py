"""
geoai/config.py
Centralized configuration for the GeoAI REE Prospectivity Engine.
"""
import os
from pathlib import Path

# Project root (where app.py lives)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Default data directories (within project root for portability)
DEPOSITS_FOLDER = Path(os.getenv("GEOAI_DEPOSITS_DIR", PROJECT_ROOT / "data" / "deposits"))
OUTPUT_DIR      = Path(os.getenv("GEOAI_OUTPUT_DIR", PROJECT_ROOT / "outputs"))
WATCH_FOLDER    = DEPOSITS_FOLDER
GDRIVE_FOLDER   = Path(os.getenv("GEOAI_GDRIVE_DIR", PROJECT_ROOT / "storage" / "gdrive"))
LOG_FILE        = OUTPUT_DIR / "watcher.log"

# Create directories if they don't exist
for path in [DEPOSITS_FOLDER, OUTPUT_DIR, GDRIVE_FOLDER]:
    path.mkdir(parents=True, exist_ok=True)

# Application constants
COOLDOWN_SECS = 120
VERSION = "1.0.0"

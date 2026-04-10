"""
scripts/reset_system.py
Safely wipes all training data, outputs, and synced storage to provide a clean slate for Geo-AI-India.
"""
import shutil
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))
from geoai.config import DEPOSITS_FOLDER, OUTPUT_DIR, GDRIVE_FOLDER

def reset():
    print("="*50)
    print("  Geo-AI-India: SYSTEM RESET")
    print("="*50)
    
    targets = [
        ("Raw Deposits Data", DEPOSITS_FOLDER),
        ("Model Outputs & Results", OUTPUT_DIR),
        ("G-Drive Sync Folder", GDRIVE_FOLDER),
    ]
    
    for label, path in targets:
        if path.exists():
            print(f"  Cleaning {label}: {path}")
            # Instead of deleting the folder itself, we delete its contents
            # to keep the folder structure intact.
            for item in path.iterdir():
                try:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
                    print(f"    - Deleted: {item.name}")
                except Exception as e:
                    print(f"    - ERROR deleting {item.name}: {e}")
        else:
            print(f"  [SKIP] {label} not found.")

    print("\n  System reset complete. Ready for new data ingestion.")
    print("="*50)

if __name__ == "__main__":
    reset()

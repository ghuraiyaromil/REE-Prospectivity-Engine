"""
REE ENGINE -- STEP 0: EXTRACT ARCHIVES
Extracts ALL .tar and .zip files from both data folders.
"""
import sys, tarfile, zipfile, datetime
from pathlib import Path

# ── BOTH folders are scanned for archives ─────────────
FOLDERS_TO_SCAN = [
    r"D:\GeoAI-INDIA\training_data",
    r"D:\GeoAI-INDIA\training_data_extracted",
]
EXTRACT_TO = r"D:\GeoAI-INDIA\training_data_extracted"
# ──────────────────────────────────────────────────────

def human_size(b):
    for u in ['B','KB','MB','GB']:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"

out = Path(EXTRACT_TO)
out.mkdir(parents=True, exist_ok=True)

archives = []
for folder in FOLDERS_TO_SCAN:
    p = Path(folder)
    if not p.exists():
        print(f"  [SKIP] Not found: {folder}")
        continue
    for ext in ['*.tar','*.tar.gz','*.tgz','*.zip','*.tar.bz2']:
        archives += list(p.rglob(ext))

if not archives:
    print("\n  No archives found in any folder.")
    input("  Press Enter to exit..."); sys.exit(0)

print(f"\n  Found {len(archives)} archive(s) to extract")
total = 0
for i, arc in enumerate(archives, 1):
    sz = human_size(arc.stat().st_size)
    print(f"\n  [{i}/{len(archives)}] {arc.name}  ({sz})")
    arc_out = out / arc.stem.replace('.tar','')
    arc_out.mkdir(exist_ok=True)
    try:
        if arc.suffix == '.zip':
            with zipfile.ZipFile(arc,'r') as zf:
                members = zf.namelist()
                print(f"    {len(members)} files inside...")
                zf.extractall(arc_out)
        else:
            with tarfile.open(arc,'r:*') as tf:
                members = tf.getnames()
                print(f"    {len(members)} files inside...")
                tf.extractall(arc_out)
        total += len(members)
        print(f"    Done -> {arc_out}")
    except Exception as e:
        print(f"    ERROR: {e}")

print(f"\n  Extracted {total:,} files total -> {EXTRACT_TO}")
print(f"  Now run RUN_STEP1_INVENTORY.bat")
input("\n  Press Enter to close...")

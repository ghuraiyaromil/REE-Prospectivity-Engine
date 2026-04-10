"""
REE ENGINE -- STEP 0: EXTRACT ARCHIVES
Extracts ALL .tar and .zip files from the deposits folder.
"""
import sys
import tarfile
import zipfile
from pathlib import Path

# Add project root to path for geoai import
sys.path.append(str(Path(__file__).parent.parent))
from geoai.config import DEPOSITS_FOLDER

# ── Configuration ─────────────────────────────────────────────
FOLDERS_TO_SCAN = [DEPOSITS_FOLDER]
EXTRACT_TO = DEPOSITS_FOLDER / "extracted"
# ──────────────────────────────────────────────────────────────


def human_size(b):
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def main():
    out = Path(EXTRACT_TO)
    out.mkdir(parents=True, exist_ok=True)

    archives = []
    for folder in FOLDERS_TO_SCAN:
        p = Path(folder)
        if not p.exists():
            print(f"  [SKIP] Not found: {folder}")
            continue
        for ext in ["*.tar", "*.tar.gz", "*.tgz", "*.zip", "*.tar.bz2"]:
            archives += list(p.rglob(ext))

    if not archives:
        print("\n  No archives found in any folder.")
        return

    print(f"\n  Found {len(archives)} archive(s) to extract")
    total = 0
    for i, arc in enumerate(archives, 1):
        sz = human_size(arc.stat().st_size)
        print(f"\n  [{i}/{len(archives)}] {arc.name}  ({sz})")
        arc_out = out / arc.stem.replace(".tar", "")
        arc_out.mkdir(exist_ok=True)
        try:
            if arc.suffix == ".zip":
                with zipfile.ZipFile(arc, "r") as zf:
                    members = zf.namelist()
                    print(f"    {len(members)} files inside...")
                    zf.extractall(arc_out)
            else:
                with tarfile.open(arc, "r:*") as tf:
                    members = tf.getnames()
                    print(f"    {len(members)} files inside...")
                    tf.extractall(arc_out)
            total += len(members)
            print(f"    Done -> {arc_out}")
        except Exception as e:
            print(f"    ERROR: {e}")

    print(f"\n  Extracted {total:,} files total -> {EXTRACT_TO}")
    print("  Now run step1_inventory.py")


if __name__ == "__main__":
    main()

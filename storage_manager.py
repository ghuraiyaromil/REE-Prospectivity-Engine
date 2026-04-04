"""
=================================================================
  GeoAI -- STORAGE MANAGER
  storage_manager.py

  Run this AFTER the pipeline has trained on a deposit.
  It verifies features are saved, then tells you exactly
  which files are safe to delete to free disk space.

  USAGE:
    python storage_manager.py --deposit mount_weld
    python storage_manager.py --check_all
    python storage_manager.py --free mount_weld --confirm
=================================================================
"""
import argparse, json, shutil
from pathlib import Path

DEPOSITS_FOLDER = r"D:\GeoAI-INDIA\deposits"
OUTPUT_FOLDER   = r"D:\GeoAI-INDIA\ree_output"

# Extensions that are safe to delete after feature extraction
RASTER_EXTENSIONS = {
    ".tif",".tiff",".ers",".img",".ecw",".jp2",".nc",
    ".hdf",".h5",".asc",".grd",".gxf",".bil",
}
# Extensions that must ALWAYS be kept
KEEP_FOREVER = {".csv",".xlsx",".xls",".txt",".json",".shp",
                ".dbf",".shx",".prj",".kml",".geojson"}
# Archives can be deleted after extraction
ARCHIVE_EXT = {".tar",".zip",".gz",".tgz",".bz2"}


def human_size(b):
    for u in ["B","KB","MB","GB"]:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def scan_deposit(deposit_folder):
    """Return summary of what's in a deposit folder."""
    p = Path(deposit_folder)
    if not p.exists():
        return None

    deletable = []   # rasters and archives — safe to delete after training
    keep      = []   # CSVs and small files — keep forever

    for f in p.rglob("*"):
        if not f.is_file(): continue
        ext = f.suffix.lower()
        size = f.stat().st_size
        if ext in RASTER_EXTENSIONS or ext in ARCHIVE_EXT:
            deletable.append((f, size))
        elif ext in KEEP_FOREVER:
            keep.append((f, size))

    return {
        "deletable": deletable,
        "keep":      keep,
        "deletable_bytes": sum(s for _,s in deletable),
        "keep_bytes":      sum(s for _,s in keep),
    }


def feature_matrix_exists(deposit_name):
    """Check if feature matrix was saved for this deposit."""
    out = Path(OUTPUT_FOLDER)
    # Look for scored CSV or feature matrix
    patterns = [
        f"scored_{deposit_name}*.csv",
        f"feature_matrix_{deposit_name}*.csv",
        "scored_data.csv",
    ]
    for pat in patterns:
        if list(out.glob(pat)):
            return True
    # Check registry
    reg_path = out / "deposit_registry.json"
    if reg_path.exists():
        reg = json.loads(reg_path.read_text())
        if deposit_name in reg.get("deposits", {}):
            return True
    return False


def check_all():
    """Show storage summary for all deposits."""
    deposits_path = Path(DEPOSITS_FOLDER)
    if not deposits_path.exists():
        print(f"  Deposits folder not found: {DEPOSITS_FOLDER}")
        return

    print()
    print("="*62)
    print("  GeoAI STORAGE REPORT")
    print("="*62)

    total_deletable = 0
    total_keep      = 0

    for dep_folder in sorted(deposits_path.iterdir()):
        if not dep_folder.is_dir(): continue
        name   = dep_folder.name
        result = scan_deposit(dep_folder)
        if not result: continue

        trained = feature_matrix_exists(name)
        status  = "[TRAINED]" if trained else "[NOT TRAINED YET]"

        print()
        print(f"  {name.upper()}  {status}")
        print(f"  Folder: {dep_folder}")
        print(f"  Keep forever (CSVs etc): {human_size(result['keep_bytes'])}")
        print(f"  Can delete after training: {human_size(result['deletable_bytes'])}")

        if trained:
            print(f"  --> SAFE TO FREE {human_size(result['deletable_bytes'])} now")
            for f, s in result["deletable"][:5]:
                print(f"      {f.name}  ({human_size(s)})")
            if len(result["deletable"]) > 5:
                print(f"      ... and {len(result['deletable'])-5} more files")
        else:
            print(f"  --> Run pipeline first before deleting anything")

        total_deletable += result["deletable_bytes"]
        total_keep      += result["keep_bytes"]

    print()
    print("="*62)
    print(f"  TOTAL you can free right now:  {human_size(total_deletable)}")
    print(f"  TOTAL you must keep forever:   {human_size(total_keep)}")
    print("="*62)


def free_deposit(deposit_name, confirm=False):
    """Delete rasters and archives for a trained deposit."""
    dep_folder = Path(DEPOSITS_FOLDER) / deposit_name

    if not dep_folder.exists():
        print(f"  Deposit folder not found: {dep_folder}")
        return

    if not feature_matrix_exists(deposit_name):
        print(f"  ERROR: No trained model found for '{deposit_name}'")
        print(f"  Run the pipeline first before deleting raw data!")
        return

    result = scan_deposit(dep_folder)
    if not result["deletable"]:
        print(f"  Nothing to delete for {deposit_name} — already clean.")
        return

    print()
    print(f"  Deposit: {deposit_name}")
    print(f"  Files to delete: {len(result['deletable'])}")
    print(f"  Space to free:   {human_size(result['deletable_bytes'])}")
    print()
    print("  Files that will be DELETED:")
    for f, s in result["deletable"]:
        print(f"    {f.relative_to(dep_folder)}  ({human_size(s)})")
    print()
    print("  Files that will be KEPT:")
    for f, s in result["keep"]:
        print(f"    {f.relative_to(dep_folder)}  ({human_size(s)})")

    if not confirm:
        print()
        print("  To actually delete, run:")
        print(f"  python storage_manager.py --free {deposit_name} --confirm")
        return

    print()
    print("  Deleting...")
    freed = 0
    for f, s in result["deletable"]:
        try:
            f.unlink()
            freed += s
            print(f"    Deleted: {f.name}")
        except Exception as e:
            print(f"    Could not delete {f.name}: {e}")

    # Remove empty subdirectories
    for sub in sorted(dep_folder.rglob("*"), reverse=True):
        if sub.is_dir():
            try:
                sub.rmdir()   # only removes if empty
            except Exception:
                pass

    print()
    print(f"  Freed: {human_size(freed)}")
    print(f"  Kept:  {human_size(result['keep_bytes'])} (CSVs — permanent)")


def main():
    parser = argparse.ArgumentParser(
        description="GeoAI storage manager — free disk space after training"
    )
    parser.add_argument("--check_all", action="store_true",
                        help="Show storage summary for all deposits")
    parser.add_argument("--free", metavar="DEPOSIT",
                        help="Free raster space for a trained deposit")
    parser.add_argument("--deposit", metavar="DEPOSIT",
                        help="Check a single deposit")
    parser.add_argument("--confirm", action="store_true",
                        help="Actually delete (without this, just shows what would be deleted)")
    args = parser.parse_args()

    if args.check_all or (not args.free and not args.deposit):
        check_all()
    elif args.deposit:
        result = scan_deposit(Path(DEPOSITS_FOLDER)/args.deposit)
        trained = feature_matrix_exists(args.deposit)
        if result:
            print(f"\n  {args.deposit}: "
                  f"{'TRAINED' if trained else 'not trained'}")
            print(f"  Deletable: {human_size(result['deletable_bytes'])}")
            print(f"  Keep:      {human_size(result['keep_bytes'])}")
    elif args.free:
        free_deposit(args.free, confirm=args.confirm)

    input("\n  Press Enter to close...")

if __name__ == "__main__":
    main()

"""
REE ENGINE -- STEP 1: FULL DATA INVENTORY
Scans deposit folders and produces a detailed report.
"""
import os
import sys
import json
import datetime
from pathlib import Path
from collections import defaultdict

# Add project root to path for geoai import
sys.path.append(str(Path(__file__).parent.parent))
from geoai.config import DEPOSITS_FOLDER, OUTPUT_DIR

# ── Configuration ─────────────────────────────────────────────
DATA_FOLDERS = [DEPOSITS_FOLDER, DEPOSITS_FOLDER / "extracted"]
REPORT_FOLDER = OUTPUT_DIR
# ──────────────────────────────────────────────────────────────


def human_size(b):
    for u in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"


def categorise(ext):
    cats = {
        "tabular":  [".csv", ".txt", ".tsv", ".xlsx", ".xls"],
        "gis":      [".shp", ".gpkg", ".geojson", ".kml", ".gdb"],
        "raster":   [".tif", ".tiff", ".asc", ".grd", ".ers", ".img", ".nc", ".hdf", ".h5"],
        "database": [".db", ".sqlite", ".mdb"],
        "document": [".pdf", ".doc", ".docx"],
        "image":    [".jpg", ".jpeg", ".png", ".ecw", ".jp2"],
        "geophys":  [".dat", ".xyz", ".gxf"],
        "data":     [".json", ".npy", ".pkl", ".parquet"],
        "archive":  [".tar", ".zip", ".gz", ".bz2", ".tgz"],
    }
    for cat, exts in cats.items():
        if ext in exts:
            return cat
    return "other"


REE_KW = [
    "ce", "la", "nd", "pr", "sm", "eu", "treo", "tree", "ree",
    "lree", "hree", "ceo2", "la2o3", "rare earth", "rare_earth",
]


def main():
    lines = []

    def pr(s):
        print(f"  {s}")
        lines.append(s)

    print("\n" + "=" * 62)
    print("  REE ENGINE -- DATA INVENTORY (multi-folder)")
    print("=" * 62 + "\n")

    grand = defaultdict(lambda: {"count": 0, "size": 0, "exts": set()})
    all_paths = defaultdict(list)
    all_ree = []
    all_cols = {}
    all_json_info = []
    folder_summaries = []

    for folder in DATA_FOLDERS:
        p = Path(folder)
        if not p.exists():
            print(f"  [SKIP] Not found: {folder}")
            continue

        print(f"  Scanning: {folder} ...")
        f_count = 0
        f_size = 0
        ext_groups = defaultdict(list)

        for f in p.rglob("*"):
            if not f.is_file():
                continue
            size = f.stat().st_size
            f_count += 1
            f_size += size
            ext = f.suffix.lower()
            ext_groups[ext].append({"name": f.name, "path": str(f), "size": size})

        folder_summaries.append((str(folder), f_count, f_size))

        for ext, file_infos in ext_groups.items():
            cat = categorise(ext)
            grand[cat]["count"] += len(file_infos)
            grand[cat]["size"] += sum(fi["size"] for fi in file_infos)
            grand[cat]["exts"].add(ext or "(none)")
            all_paths[ext] += [fi["path"] for fi in file_infos]

        # Peek inside CSVs
        for info in ext_groups.get(".csv", [])[:15]:
            try:
                with open(info["path"], "r", errors="ignore") as fh:
                    hdr = fh.readline().strip().lower()
                    cols = [c.strip().strip('"') for c in hdr.split(",")][:25]
                    all_cols[info["name"]] = cols
                    for col in cols:
                        if any(k in col for k in REE_KW):
                            all_ree.append(f"{info['name']} -> column: {col}")
            except Exception:
                pass

        # Peek inside JSONs
        for info in ext_groups.get(".json", [])[:5]:
            try:
                with open(info["path"], "r", errors="ignore") as fh:
                    obj = json.load(fh)
                    if isinstance(obj, dict):
                        all_json_info.append(
                            f"{info['name']}: keys={list(obj.keys())[:6]}"
                        )
                    elif isinstance(obj, list) and obj:
                        first_keys = (
                            list(obj[0].keys())[:5]
                            if isinstance(obj[0], dict)
                            else type(obj[0]).__name__
                        )
                        all_json_info.append(
                            f"{info['name']}: list[{len(obj)}] first={first_keys}"
                        )
            except Exception:
                pass

    # ── Build report ─────────────────────────────────────────
    sep = "=" * 62
    pr(sep)
    pr("  REE PROSPECTIVITY ENGINE -- DATA INVENTORY REPORT")
    pr(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    pr(sep)
    pr("")

    pr("FOLDERS SCANNED")
    pr("---------------")
    grand_files = grand_size = 0
    for folder_str, fc, fs in folder_summaries:
        pr(f"  {folder_str}")
        pr(f"    Files: {fc:,}   Size: {human_size(fs)}")
        grand_files += fc
        grand_size += fs
    pr("")
    pr(f"COMBINED TOTALS: {grand_files:,} files   {human_size(grand_size)}")
    pr("")

    pr("FILE TYPES BY CATEGORY")
    pr("----------------------")
    priority = ["tabular", "data", "raster", "gis", "geophys", "image",
                "database", "document", "archive", "other"]
    for cat in priority:
        if cat in grand:
            info = grand[cat]
            exts = ", ".join(sorted(info["exts"]))
            pr(f"  {cat.upper():<12}  {info['count']:>6} files  "
               f"{human_size(info['size']):>9}   [{exts}]")
    pr("")

    if all_ree:
        pr("REE / GEOCHEMICAL COLUMNS DETECTED")
        pr("-----------------------------------")
        for h in all_ree[:40]:
            pr(f"  [FOUND] {h}")
        pr("")

    if all_cols:
        pr("CSV COLUMN HEADERS (sample)")
        pr("---------------------------")
        for fname, cols in list(all_cols.items())[:12]:
            non_empty = [c for c in cols if c.strip()]
            pr(f"  {fname}")
            pr(f"    {', '.join(non_empty[:20])}{'...' if len(non_empty) > 20 else ''}")
        pr("")

    pr(sep)

    report_txt = "\n".join(lines)
    # Save report
    for loc in [str(REPORT_FOLDER), str(Path(__file__).parent)]:
        try:
            rp = Path(loc) / "inventory_report.txt"
            rp.parent.mkdir(parents=True, exist_ok=True)
            rp.write_text(report_txt, encoding="utf-8")
            print(f"\n  Report saved: {rp}")
            break
        except Exception:
            pass

    # Save JSON index
    try:
        idx = Path(__file__).parent / "inventory_data.json"
        idx.write_text(
            json.dumps({
                "ext_paths": {k: v[:30] for k, v in all_paths.items()},
                "csv_cols":  all_cols,
                "ree_hints": all_ree,
            }, indent=2),
            encoding="utf-8",
        )
        print(f"  Data index: {idx}")
    except Exception:
        pass

    print("\n  Done!")


if __name__ == "__main__":
    main()

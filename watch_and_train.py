"""
=================================================================
  GeoAI -- AUTOMATIC TRAINING WATCHER
  watch_and_train.py

  DROP a new deposit folder anywhere under WATCH_FOLDER.
  This script detects it, runs the full pipeline, and uploads
  the trained model bundle to Google Drive automatically.

  HOW TO USE:
    1. Install:   pip install watchdog pydrive2
    2. Run:       python watch_and_train.py
    3. Add data:  drop a folder into D:\GeoAI-INDIA\deposits\
                  e.g. D:\GeoAI-INDIA\deposits\browns_range\
                        containing collar.csv, assay.csv, etc.
    4. Done:      pipeline trains automatically, bundle syncs to Drive

  RUNS IN BACKGROUND -- minimise the window, it keeps watching.
=================================================================
"""
import time, logging, subprocess, sys, json, shutil
from pathlib import Path
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────
from geoai.config import WATCH_FOLDER, OUTPUT_DIR, GDRIVE_FOLDER, LOG_FILE, COOLDOWN_SECS
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s  %(message)s",
    datefmt  = "%H:%M:%S",
    handlers = [
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger("GeoAI-Watcher")

def find_deposit_files(deposit_folder):
    """Scan a deposit folder and return all data files."""
    p = Path(deposit_folder)
    files = []
    for ext in ["*.csv","*.xlsx","*.xls","*.tif","*.tiff","*.shp",
                "*.ers","*.img","*.jp2","*.ecw","*.asc","*.xyz",
                "*.json","*.geojson","*.kml","*.zip","*.tar"]:
        files.extend(p.rglob(ext))
    return files

def infer_deposit_name(folder_path):
    """Use folder name as deposit name, normalised."""
    name = Path(folder_path).name.lower()
    name = name.replace(" ","_").replace("-","_")
    return name

def run_pipeline(deposit_folder, deposit_name):
    """Run the GeoAI pipeline on a deposit folder."""
    log.info(f"  Starting pipeline for: {deposit_name}")
    try:
        # Import and run pipeline directly
        sys.path.insert(0, str(Path(__file__).parent))
        from geoai.pipeline import GeoAIPipeline

        files = find_deposit_files(deposit_folder)
        if not files:
            log.warning(f"  No data files found in {deposit_folder}")
            return False

        log.info(f"  Found {len(files)} files")

        pipe   = GeoAIPipeline(output_dir=OUTPUT_DIR)
        result = pipe.run(
            files        = files,
            deposit_name = deposit_name,
            progress_cb  = lambda msg: log.info(f"    {msg}"),
        )

        if result.get("status") == "success":
            log.info(f"  Pipeline complete:")
            log.info(f"    CV R²    = {result['cv_r2']:.4f}")
            log.info(f"    ROC AUC  = {result['roc_auc']:.4f}")
            log.info(f"    Holes    = {result['n_labelled']}")
            log.info(f"    Features = {result['n_features']}")
            return True
        else:
            log.warning(f"  Pipeline returned: {result.get('status')}")
            return False

    except Exception as e:
        log.error(f"  Pipeline error: {e}")
        import traceback
        log.error(traceback.format_exc())
        return False

def sync_to_gdrive():
    """Copy model bundle and registry to Google Drive folder."""
    gdrive = Path(GDRIVE_FOLDER)
    if not gdrive.exists():
        try:
            gdrive.mkdir(parents=True)
        except Exception:
            log.warning(f"  Google Drive folder not found: {GDRIVE_FOLDER}")
            log.warning(f"  Skipping sync — model saved locally only")
            return

    out = Path(OUTPUT_DIR)
    synced = []

    # Find latest bundle
    bundles = sorted(out.glob("ree_model_bundle*.joblib"),
                     key=lambda f: f.stat().st_mtime)
    if bundles:
        dest = gdrive / "ree_model_bundle.joblib"
        shutil.copy2(str(bundles[-1]), str(dest))
        synced.append("ree_model_bundle.joblib")

    # Registry and small CSVs
    for fname in ["deposit_registry.json"]:
        src = out / fname
        if src.exists():
            shutil.copy2(str(src), str(gdrive / fname))
            synced.append(fname)

    if synced:
        log.info(f"  Synced to Google Drive: {synced}")

def write_status(status, deposit=None, metrics=None):
    """Write current status to a JSON file for the app to read."""
    status_path = Path(OUTPUT_DIR) / "watcher_status.json"
    data = {
        "status":     status,
        "last_update":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "deposit":    deposit,
        "metrics":    metrics or {},
    }
    status_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

# ── MAIN WATCHER LOOP ─────────────────────────────────────────
def main():
    watch_path = Path(WATCH_FOLDER)
    watch_path.mkdir(parents=True, exist_ok=True)

    log.info("="*55)
    log.info("  GeoAI Auto-Training Watcher started")
    log.info("="*55)
    log.info(f"  Watching: {WATCH_FOLDER}")
    log.info(f"  Output:   {OUTPUT_DIR}")
    log.info(f"  Drive:    {GDRIVE_FOLDER}")
    log.info(f"  Drop a deposit folder here to trigger training")
    log.info("="*55)

    # Track already-processed deposits
    processed = {}   # deposit_name → last modified time
    last_train = 0

    try:
        while True:
            time.sleep(10)   # check every 10 seconds

            # Scan for deposit folders
            for item in watch_path.iterdir():
                if not item.is_dir(): continue
                deposit_name = infer_deposit_name(item)

                # Get latest modification time of any file in folder
                files = find_deposit_files(item)
                if not files: continue
                latest_mtime = max(f.stat().st_mtime for f in files)

                # Skip if already processed this version
                if processed.get(deposit_name) == latest_mtime:
                    continue

                # Cooldown check
                if time.time() - last_train < COOLDOWN_SECS:
                    remaining = int(COOLDOWN_SECS - (time.time()-last_train))
                    log.info(f"  Cooldown: {remaining}s remaining")
                    continue

                log.info("")
                log.info("="*55)
                log.info(f"  NEW/CHANGED deposit detected: {deposit_name}")
                log.info(f"  Folder: {item}")
                log.info(f"  Files:  {len(files)}")
                log.info("="*55)

                write_status("training", deposit=deposit_name)
                success = run_pipeline(str(item), deposit_name)
                last_train = time.time()

                if success:
                    processed[deposit_name] = latest_mtime
                    sync_to_gdrive()
                    write_status("ready", deposit=deposit_name,
                                 metrics={"last_deposit": deposit_name})
                    log.info(f"  Done. Model updated with {deposit_name}.")
                    log.info(f"  Watching for next deposit...")
                else:
                    write_status("error", deposit=deposit_name)
                    log.error(f"  Training failed for {deposit_name}")

    except KeyboardInterrupt:
        log.info("\n  Watcher stopped by user.")
        write_status("stopped")

if __name__ == "__main__":
    main()

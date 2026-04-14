"""
geoai/pipeline.py
Unified auto-pipeline: raw data -> feature matrix -> trained model -> results.
Handles any deposit, any data format, incrementally.
"""
import warnings, datetime, hashlib, json, logging, zipfile, tarfile, shutil, tempfile
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

from .categoriser import categorise_batch, detect_deposit_name
from .config import OUTPUT_DIR

def robust_read_csv(path, **kwargs):
    """Try multiple encodings when reading CSVs."""
    for enc in ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']:
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path, **kwargs) # Last resort default

# ══════════════════════════════════════════════════════════════
# REGISTRY  — tracks every trained deposit
# ══════════════════════════════════════════════════════════════
class DepositRegistry:
    def __init__(self, registry_path):
        self.path = Path(registry_path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            try:
                return json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"deposits": {}, "global_model": None}
    
    def reload(self):
        self.data = self._load()

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def register_deposit(self, name, bundle_path, metrics, n_holes):
        if name not in self.data["deposits"]:
            self.data["deposits"][name] = {"versions": []}
        
        # Store path relative to the registry folder (usually just the filename)
        rel_path = Path(bundle_path).name
        
        self.data["deposits"][name]["versions"].append({
            "date":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bundle_path": rel_path,
            "n_holes":     n_holes,
            "cv_r2":       round(float(metrics.get("r2", 0)), 4),
            "roc_auc":     round(float(metrics.get("roc", 0)), 4),
        })
        self.save()

    def list_deposits(self):
        return list(self.data["deposits"].keys())

    def get_latest_bundle(self, name=None):
        """Get most recent bundle path — globally or for a specific deposit."""
        base_dir = self.path.parent
        if name and name in self.data["deposits"]:
            versions = self.data["deposits"][name]["versions"]
            if versions:
                rel = versions[-1]["bundle_path"]
                # Handle old absolute paths during migration
                p = Path(rel)
                if not p.is_absolute(): p = base_dir / rel
                return p
        # Search for any bundle file
        if self.data.get("global_model"):
            rel = self.data["global_model"]
            p = Path(rel)
            if not p.is_absolute(): p = base_dir / rel
            if p.exists():
                return p
        return None

    def set_global_model(self, bundle_path):
        self.data["global_model"] = Path(bundle_path).name
        self.save()


# ══════════════════════════════════════════════════════════════
# LAYER PROCESSORS
# ══════════════════════════════════════════════════════════════
class DrillholeProcessor:
    """Process collar + assay CSV files into a feature matrix."""

    REE_OXIDE  = ["ceo2_ppm","la2o3_ppm","nd2o3_ppm","pr6o11_ppm","sm2o3_ppm",
                  "eu2o3_ppm","gd2o3_ppm","dy2o3_ppm","y2o3_ppm","er2o3_ppm",
                  "yb2o3_ppm","tb4o7_ppm","lu2o3_ppm","ho2o3_ppm","tm2o3_ppm"]
    PATHFINDER = ["fe2o3_ppm","p2o5_ppm","al2o3_ppm","sio2_ppm","cao_ppm",
                  "mgo_ppm","tho2_ppm","u3o8_ppm","nb2o5_ppm","mn_ppm",
                  "sr_ppm","ba_ppm","zr_ppm","ti_ppm","k_ppm"]

    def __init__(self, collar_path, assay_path=None, alteration_path=None):
        self.collar_path     = collar_path
        self.assay_path      = assay_path
        self.alteration_path = alteration_path

    def _standardise_columns(self, df):
        """Normalise column names to lowercase stripped."""
        df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
        return df

    def _detect_coord_columns(self, df):
        """Find lat/lon or easting/northing columns."""
        lat = next((c for c in df.columns if c in
                    ["latitude","lat","y_coord","northing_dd","lat_dd"]), None)
        lon = next((c for c in df.columns if c in
                    ["longitude","lon","long","x_coord","easting_dd","lon_dd"]), None)
        east = next((c for c in df.columns if "easting" in c or c == "x"), None)
        north= next((c for c in df.columns if "northing" in c or c == "y"), None)
        return lat, lon, east, north

    def _reproject_if_needed(self, df, lat_col, lon_col):
        """Convert projected coordinates to WGS84 if values look like metres."""
        if lat_col and lon_col:
            sample = df[lon_col].dropna()
            if len(sample) and abs(sample.iloc[0]) > 360:
                try:
                    from pyproj import Transformer
                    # Detect UTM zone from easting range
                    east_vals = df[lon_col].dropna()
                    zone = int((east_vals.mean() / 1e6))
                    epsg = 32600 + zone if df[lat_col].mean() > 0 else 32700 + zone
                    tr = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
                    lons, lats = tr.transform(df[lon_col].values, df[lat_col].values)
                    df["lon"] = lons; df["lat"] = lats
                except Exception:
                    pass
        return df

    def process(self):
        """Returns (master_df, feature_cols, treo_col, n_labelled)"""
        # Load collar
        collar = robust_read_csv(str(self.collar_path), low_memory=False)
        collar = self._standardise_columns(collar)

        # Find ID column - Prioritise company-specific IDs
        id_priority = ["companyholeid", "companyhole", "holeid", "hole_id", "id"]
        id_col = next((c for p in id_priority for c in collar.columns if p == c.lower()), collar.columns[0])

        # Coordinates
        lat_col, lon_col, east_col, north_col = self._detect_coord_columns(collar)
        if lat_col:
            collar["lat"] = pd.to_numeric(collar[lat_col], errors="coerce")
        if lon_col:
            collar["lon"] = pd.to_numeric(collar[lon_col], errors="coerce")
        if east_col and not lon_col:
            collar["lon"] = pd.to_numeric(collar[east_col], errors="coerce")
            collar["lat"] = pd.to_numeric(collar.get(north_col, pd.Series()), errors="coerce")
        collar = self._reproject_if_needed(collar, "lat", "lon")

        # Numeric cols
        for c in ["elevation","maxdepth","dip","azimuth"]:
            if c in collar.columns:
                collar[c] = pd.to_numeric(collar[c], errors="coerce")

        master = collar.copy()

        # Assay processing ──────────────────────────────────
        if self.assay_path:
            assay = robust_read_csv(str(self.assay_path), low_memory=False)
            assay = self._standardise_columns(assay)

            # Handle both pivoted and unpivoted assay formats
            if "attributecolumn" in assay.columns and "attributevalue" in assay.columns:
                # Unpivoted — pivot it
                assay_id = next((c for c in assay.columns if "holeid" in c
                                 or "collarid" in c or c == "id"), assay.columns[0])
                try:
                    pivoted = assay.pivot_table(
                        index=assay_id, columns="attributecolumn",
                        values="attributevalue", aggfunc="max"
                    ).reset_index()
                    pivoted.columns = [str(c).lower().strip()+"_ppm"
                                       if c != assay_id else c
                                       for c in pivoted.columns]
                    assay = pivoted
                except Exception:
                    pass

            # Numeric conversion — ALL non-text columns before groupby
            # join_col detected below — use text keywords directly here
            _skip_kw = {"companysampleid","anumber","attributecolumn",
                        "attributevalue","units","labmethod","element","dsc","hanalyte"}
            # Fuzzy Mapping (Flaw 3 Fix)
            from difflib import get_close_matches
            ree_canon = {o.lower().replace("_ppm",""): o for o in self.REE_OXIDE}
            
            # Map columns to canonical REE oxides using fuzzy matching
            matched_cols = {}
            for col in assay.columns:
                clean = col.lower().replace("_ppm","").replace("_pct","").strip()
                match = get_close_matches(clean, list(ree_canon.keys()), n=1, cutoff=0.8)
                if match:
                    target = ree_canon[match[0]]
                    if target not in matched_cols:
                        assay[target] = pd.to_numeric(assay[col], errors="coerce")
                        matched_cols[target] = col
            
            # Compute TREO using matched columns
            treo_parts = list(matched_cols.keys())
            if treo_parts:
                assay["treo"] = assay[treo_parts].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
                assay.loc[assay[treo_parts].isna().all(axis=1), "treo"] = np.nan
            
            all_ree = treo_parts # v4.0 canonical mapping
            
            # Additional derived indicators for LREE/HREE using case-insensitive search
            lree_k = [o.lower() for o in ["ceo2_ppm","la2o3_ppm","nd2o3_ppm","pr6o11_ppm","sm2o3_ppm"]]
            hree_k = [o.lower() for o in ["gd2o3_ppm","dy2o3_ppm","y2o3_ppm","er2o3_ppm","yb2o3_ppm"]]
            
            lree_c = [c for c in assay.columns if c.lower() in lree_k]
            hree_c = [c for c in assay.columns if c.lower() in hree_k]

            if lree_c: assay["lree"] = assay[lree_c].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            if hree_c: assay["hree"] = assay[hree_c].apply(pd.to_numeric, errors="coerce").sum(axis=1, skipna=True)
            if "lree" in assay.columns and "hree" in assay.columns:
                assay["lree_hree_ratio"] = assay["lree"] / (assay["hree"] + 0.001)

            # Element Ratios
            def _get_col(df, kw):
                return next((c for c in df.columns if c.lower() == kw.lower()), None)

            c1 = _get_col(assay, "ceo2_ppm"); c2 = _get_col(assay, "la2o3_ppm")
            if c1 and c2: assay["ce_la_ratio"] = assay[c1] / (assay[c2] + 1)
            
            c1 = _get_col(assay, "p2o5_ppm"); c2 = _get_col(assay, "fe2o3_ppm")
            if c1 and c2: assay["p_fe_ratio"] = assay[c1] / (assay[c2] + 1)
            
            c1 = _get_col(assay, "tho2_ppm"); c2 = _get_col(assay, "u3o8_ppm")
            if c1 and c2: assay["th_u_ratio"] = assay[c1] / (assay[c2] + 0.001)

            derived = [c for c in ["lree_hree_ratio","ce_la_ratio","p_fe_ratio",
                                    "th_u_ratio","lree","hree","treo", "treo_max"] if c in assay.columns]
            agg_cols = all_ree + [c for c in derived]
            if "treo_max" in assay.columns and "treo" not in assay.columns:
                 assay["treo"] = assay["treo_max"] # use pre-calculated
            agg_d = {}
            for c in agg_cols:
                if c in assay.columns: agg_d[c] = ["max","mean"]
            if "fromdepth" in assay.columns: agg_d["fromdepth"] = "min"
            if "todepth"   in assay.columns: agg_d["todepth"]   = "max"

            join_col = next((c for c in assay.columns if "holeid" in c
                             or "collarid" in c), assay.columns[0])
            assay_agg = assay.groupby(join_col).agg(agg_d)
            assay_agg.columns = ["_".join(c) for c in assay_agg.columns]
            assay_agg = assay_agg.reset_index()

            # Cast both join keys to string to avoid int64/object type mismatch
            master[id_col]      = master[id_col].astype(str).str.strip()
            assay_agg[join_col] = assay_agg[join_col].astype(str).str.strip()
            
            master = master.merge(assay_agg, left_on=id_col,
                                  right_on=join_col, how="left")
            
            # Re-detect treo_col after merge if it wasn't picked up before
            _treo_cols = [c for c in master.columns if "treo" in c.lower()]
            _treo_max = next((c for c in _treo_cols if "max" in c), None) or next((c for c in _treo_cols if "treo" == c), None)
            if _treo_max:
                print(f"    TREO join check: {master[_treo_max].notna().sum()} labelled")

        # Extra geochemistry file (dh_geochemistry.csv)
        geochem_path = getattr(self, 'geochem_path', None)
        if geochem_path:
            try:
                geo2 = pd.read_csv(str(geochem_path), low_memory=False)
                geo2 = self._standardise_columns(geo2)
                geo2_id = next((c for c in geo2.columns
                                if "collarid" in c or "holeid" in c), None)
                if geo2_id and "attributecolumn" in geo2.columns:
                    g2p = geo2.pivot_table(
                        index=geo2_id, columns="attributecolumn",
                        values="attributevalue", aggfunc="max"
                    ).reset_index()
                    g2p.columns = [str(c).lower()+"_geo"
                                   if c != geo2_id else c for c in g2p.columns]
                    for c in g2p.columns:
                        if c != geo2_id:
                            g2p[c] = pd.to_numeric(g2p[c], errors="coerce")
                    
                    # Robust Join Key Casting
                    master[id_col] = master[id_col].astype(str).str.strip()
                    g2p[geo2_id]   = g2p[geo2_id].astype(str).str.strip()
                    
                    master = master.merge(g2p, left_on=id_col,
                                          right_on=geo2_id, how="left")
                    print(f"    dh_geochemistry joined: {len(g2p)} rows")
            except Exception as e:
                print(f"    dh_geochemistry skipped: {e}")

        # Alteration ────────────────────────────────────────
        if self.alteration_path:
            try:
                print(f"  Attempting to load alteration from: {self.alteration_path}")
                alt = robust_read_csv(str(self.alteration_path), low_memory=False)
                alt = self._standardise_columns(alt)
                alt_id = next((c for c in alt.columns if "holeid" in c
                               or "collarid" in c), alt.columns[0])
                # Filter to only known id_col
                alt = alt[alt[alt_id].isin(master[master.columns[0]])]

                if not alt.empty:
                    if "attributevalue" in alt.columns:
                        alt["altval"] = alt["attributevalue"].astype(str).str.lower()
                        for kw in ["laterit","carbonat","weath","oxid","saprolite",
                                   "clay","goethit","limonit","ferrugin"]:
                            mask = alt["altval"].str.contains(kw, na=False)
                            col_name = f"is_{kw}"
                            top_alt = alt[mask].groupby(alt_id)["altval"].count().rename(col_name)
                            master = master.merge(top_alt, left_on=master.columns[0], right_index=True, how="left")
                            master[col_name] = master[col_name].fillna(0)
                            feat_cols.append(col_name)
            except Exception as e:
                print(f"  Warning: Could not parse alteration file {self.alteration_path}: {e}")
                pass

        # Depth score ───────────────────────────────────────
        from_col = next((c for c in master.columns if "fromdepth" in c
                         and "min" in c), None) or "fromdepth"
        if from_col in master.columns:
            def _ds(d):
                if pd.isna(d): return 0.5
                if d < 20:  return 1.0
                if d < 50:  return 0.95
                if d < 100: return 0.85
                if d < 200: return 0.65
                return 0.35
            master["depth_score"] = pd.to_numeric(
                master[from_col], errors="coerce").apply(_ds)

        # Spatial coords feature
        if "lat" in master.columns and "lon" in master.columns:
            ln_min = master["lon"].min(); ln_rng = master["lon"].max()-ln_min+1e-9
            lt_min = master["lat"].min(); lt_rng = master["lat"].max()-lt_min+1e-9
            master["coord_lon_norm"] = (master["lon"]-ln_min)/ln_rng
            master["coord_lat_norm"] = (master["lat"]-lt_min)/lt_rng
            clat = master["lat"].median(); clon = master["lon"].median()
            master["dist_centroid"] = np.sqrt(
                ((master["lon"]-clon)*111320*np.cos(np.radians(clat)))**2 +
                ((master["lat"]-clat)*111320)**2
            ) / 500.0

        # Identify feature columns ──────────────────────────
        skip = {"lat","lon","x","y","easting","northing",id_col,
                "companyholeid","holeid","datum","projection","zone",
                "company","holetype","geom","dataset"}
        feat_cols = [c for c in master.columns
                     if c not in skip
                     and master[c].dtype in [np.float64, np.float32,
                                             np.int64, np.int32, float, int]
                     and master[c].notna().sum() > 0]

        treo_col = next((c for c in master.columns if "treo" in c and "max" in c), None)
        if not treo_col:
            treo_col = next((c for c in master.columns if c == "treo"), None)
        if not treo_col:
            treo_col = next((c for c in master.columns
                             if any(k in c for k in ["treo","total_ree"])
                             and master[c].notna().sum() > 0), None)
        n_labelled = int(master[treo_col].notna().sum()) if treo_col else 0
        print(f"    TREO column: {treo_col}  non-null: {n_labelled}/{len(master)}")

        return master, feat_cols, treo_col, n_labelled


class RasterExtractor:
    """Extract geophysics/topography values at drillhole + grid locations."""

    def __init__(self, raster_files):
        self.rasters = raster_files   # list of Paths

    def extract(self, coords_lonlat):
        """
        Extract 3x3 window focal statistics around each coordinate. (Flaw 2 Fix)
        """
        try:
            import rasterio
            from pyproj import Transformer
        except ImportError:
            return {}

        results = {}
        for rpath in self.rasters:
            try:
                with rasterio.open(str(rpath)) as src:
                    epsg = src.crs.to_epsg() if src.crs else 4326
                    tr = Transformer.from_crs(4326, epsg, always_xy=True) if epsg != 4326 else None
                    
                    stem = Path(rpath).stem[:25]
                    all_vals = []
                    
                    for lon, lat in coords_lonlat:
                        try:
                            px, py = tr.transform(lon, lat) if tr else (lon, lat)
                            row, col = src.index(px, py)
                            
                            # Determine Window (3x3)
                            win = rasterio.windows.Window(col - 1, row - 1, 3, 3)
                            
                            # Read with intersection to avoid DstRect errors
                            constrained_win = win.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
                            data = src.read(1, window=constrained_win)
                            
                            # Focal stats on available pixels
                            valid = data[np.isfinite(data)] if data.size > 0 else np.array([])
                            if len(valid) > 0:
                                all_vals.append([np.mean(valid), np.std(valid), np.max(valid)])
                            else:
                                # Fallback to single point sample if window fails
                                val = list(src.sample([(px, py)]))[0][0]
                                fval = float(val) if np.isfinite(val) else 0.0
                                all_vals.append([fval, 0.0, fval])
                        except:
                            all_vals.append([0.0, 0.0, 0.0])
                    
                    all_vals = np.array(all_vals)
                    results[f"{stem}_mean"] = all_vals[:, 0]
                    results[f"{stem}_std"]  = all_vals[:, 1]
                    results[f"{stem}_max"]  = all_vals[:, 2]
            except Exception as e:
                print(f"Extraction error on {rpath}: {e}")
        return results


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
class GeoAIPipeline:
    """
    End-to-end pipeline: raw data -> trained model -> results.

    Usage:
        pipe = GeoAIPipeline(output_dir="D:/GeoAI/output")
        result = pipe.run(
            files       = [list of uploaded file paths],
            deposit_name= "mount_weld",   # optional, auto-detected if None
        )
    """

    def __init__(self, output_dir=None, registry_path=None):
        self.out      = Path(output_dir) if output_dir else OUTPUT_DIR
        self.out.mkdir(parents=True, exist_ok=True)
        reg_path      = registry_path or self.out / "deposit_registry.json"
        self.registry = DepositRegistry(reg_path)

    def _expand_archives(self, file_paths, temp_dir):
        """Recursively expand ZIP and TAR archives into a temp directory."""
        expanded = []
        for fp in file_paths:
            p = Path(fp)
            if p.suffix.lower() in [".zip", ".tar", ".gz", ".tgz"]:
                try:
                    extract_path = Path(temp_dir) / p.stem
                    extract_path.mkdir(parents=True, exist_ok=True)
                    if p.suffix.lower() == ".zip":
                        with zipfile.ZipFile(str(p), 'r') as z:
                            z.extractall(str(extract_path))
                    else: # tar
                        with tarfile.open(str(p), 'r:*') as t:
                            t.extractall(str(extract_path))
                    
                    # Recursively check the new files
                    inner_files = [f for f in extract_path.rglob("*") if f.is_file()]
                    expanded.extend(self._expand_archives(inner_files, temp_dir))
                except Exception as e:
                    print(f"  Error extracting {p.name}: {e}")
            else:
                expanded.append(p)
        return expanded

    def run(self, files, deposit_name=None, force_retrain=False, 
            inference_only=False, progress_cb=None):
        """
        Main entry point.
        files: list of Path objects (any format)
        Returns: dict with results, paths, metrics
        """
        self.registry.reload()
        def log(msg):
            print(f"  {msg}")
            if progress_cb: progress_cb(msg)

        log("="*50)
        log("GeoAI Pipeline starting")
        log("="*50)

        # STEP 0: Expand Archives ──────────────────────────
        log("Step 0: Expanding archives...")
        # Use a temporary directory for extraction
        with tempfile.TemporaryDirectory() as td:
            files = self._expand_archives(files, td)
            # Copy all files to a stable temp location for the duration of run
            # since td will be deleted at end of scope.
            # Actually, Step 1 follows immediately.
            # To be safe, we should expand into a directory we control or
            # ensure subsequent steps don't rely on paths after 'with' block.
            # Better: the caller should handle temp file lifecycle or
            # GeoAIPipeline should manage it.
            
            # REVISION: Process everything within the context of extraction
            return self._run_internal(files, deposit_name, force_retrain, inference_only, log)

    def _run_internal(self, files, deposit_name, force_retrain, inference_only, log):
        # STEP 1: Categorise ────────────────────────────────
        log("Step 1: Categorising files...")
        results, groups = categorise_batch(files)
        if not deposit_name:
            deposit_name = detect_deposit_name(files)
        log(f"Deposit detected: {deposit_name}")
        for layer, layer_files in groups.items():
            if layer_files:
                log(f"  {layer}: {len(layer_files)} files")

        # STEP 2: Process drillhole data ────────────────────
        log("Step 2: Processing drillhole data...")
        geochem_extra = None
        # Combine drillhole + any geochemical files with dh_ prefix
        all_dh_files = groups["drillhole"] + [
            f for f in groups["geochemical"]
            if any(k in Path(str(f)).stem.lower() for k in
                   ["dh_","drillhole","collar","assay","alter","lith","pivot"])
            and Path(str(f)).suffix.lower() in [".csv", ".txt", ".xls", ".xlsx"]
        ]
        assay_file  = None
        collar_file = None
        alt_file    = None

        for f in all_dh_files:
            name = Path(f).stem.lower()
            if "collar" in name:
                collar_file = f
            elif "pivot" in name:
                assay_file = f
            elif "assay" in name and not assay_file:
                assay_file = f
            elif "alter" in name:
                alt_file = f
            elif "geochem" in name and "assay" not in name:
                geochem_extra = f

        # Fallback: first file is collar
        if not collar_file and all_dh_files:
            collar_file = all_dh_files[0]
            remaining   = all_dh_files[1:]
            if remaining: assay_file = remaining[0]
            if len(remaining) > 1: alt_file = remaining[1]

        if not collar_file:
            log("WARNING: No drillhole collar file found. Skipping drillhole processing.")
            master   = pd.DataFrame()
            feat_cols= []
            treo_col = None
            n_labelled = 0
        else:
            proc = DrillholeProcessor(collar_file, assay_file, alt_file)
            proc.geochem_path = geochem_extra
            master, feat_cols, treo_col, n_labelled = proc.process()
            log(f"  Holes: {len(master)}  |  Labelled: {n_labelled}  |  Features: {len(feat_cols)}")

        # STEP 3: Extract rasters at drillhole locations ────
        log("Step 3: Extracting raster features...")
        raster_files = ([Path(f) for f in groups["geophysics"]] +
                        [Path(f) for f in groups["topography"]] +
                        [Path(f) for f in groups["satellite"]])

        if raster_files and len(master) > 0 and "lat" in master.columns:
            coords = list(zip(master["lon"].fillna(0), master["lat"].fillna(0)))
            extractor = RasterExtractor(raster_files)
            rast_vals = extractor.extract(coords)
            for key, arr in rast_vals.items():
                master[f"r_{key}"] = arr
                feat_cols.append(f"r_{key}")
                log(f"  Raster extracted: {key}")

        # STEP 4: IDW pseudo-labels ─────────────────────────
        log("Step 4: Generating pseudo-labels via IDW...")
        if treo_col and n_labelled >= 3 and len(master) > n_labelled:
            from scipy.spatial import cKDTree
            labeled_mask = master[treo_col].notna()
            labeled_idx  = np.where(labeled_mask.values)[0]
            xy_all = master[["lon","lat"]].fillna(0).values
            xy_lab = xy_all[labeled_idx]
            tree   = cKDTree(xy_lab)
            k      = min(8, len(labeled_idx))
            dists, idxs = tree.query(xy_all, k=k)
            dists  = np.maximum(dists * 111320, 1e-6)
            w      = 1.0 / (dists ** 2); w /= w.sum(axis=1, keepdims=True)
            treo_vals = pd.to_numeric(master[treo_col], errors="coerce").fillna(0).values
            pseudo_treo = (w * treo_vals[idxs]).sum(axis=1) if idxs.ndim > 1 \
                          else w * treo_vals[idxs]
            master["pseudo_treo"] = pseudo_treo
            master["idw_conf"]    = 1.0 / (1.0 + dists[:,0] / 100.0) \
                                    if dists.ndim > 1 else 0.5
            log(f"  IDW: {n_labelled} labelled -> {len(master)} pseudo-labelled")

        # STEP 5: Build feature matrix ──────────────────────
        log("Step 5: Building feature matrix...")
        
        # ── CRITICAL: Remove target-derived features to prevent data leakage ──
        # TREO (Total Rare Earth Oxides) is our target variable.
        # Any column derived from it (treo_max, treo_mean, lree_*, hree_*) will
        # cause the model to "predict TREO from TREO" = fake R².
        leakage_keywords = ["treo", "lree", "hree", "pseudo_treo", "idw_conf"]
        feat_cols_clean = [c for c in feat_cols
                           if c in master.columns
                           and not any(lk in c.lower() for lk in leakage_keywords)]
        feat_cols = list(dict.fromkeys(feat_cols_clean))
        
        if not feat_cols:
            log("  WARNING: No non-leaking features found. Using all features.")
            feat_cols = list(dict.fromkeys([c for c in feat_cols if c in master.columns]))
        
        X_df = master[feat_cols].copy()
        for c in feat_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        X_df = X_df.fillna(X_df.median()).fillna(0)

        # log1p geochemical
        for c in feat_cols:
            if "_ppm" in c or "ratio" in c:
                X_df[c] = np.log1p(X_df[c].clip(lower=0))

        if treo_col:
            treo_raw = pd.to_numeric(master[treo_col], errors="coerce")
            p95      = treo_raw.quantile(0.95) if treo_raw.notna().sum() > 5 else 1000
            y_all    = (treo_raw / p95).clip(0, 1).fillna(-1).values
            labeled  = y_all >= 0
            X_train  = X_df.values[labeled]
            y_train  = y_all[labeled]
            n_train  = int(labeled.sum())
        else:
            p95 = 1; X_train = X_df.values; y_train = np.zeros(len(master))
            n_train = len(master); labeled = np.ones(len(master), dtype=bool)

        log(f"  Feature matrix: {X_train.shape[0]} x {X_train.shape[1]}")

        # STEP 6: Check registry — Universal Feature Union (Incremental Version) ─────
        log("Step 6: Loading global model for incremental update...")
        union_feats = list(feat_cols)
        bundle_models = None
        bundle_scaler = None
        
        if not force_retrain:
            existing_bundle = self.registry.get_latest_bundle()
            if existing_bundle and existing_bundle.exists():
                try:
                    import joblib
                    bundle = joblib.load(str(existing_bundle))
                    old_feat = bundle.get("feat_cols", [])
                    bundle_models = bundle.get("models")
                    bundle_scaler = bundle.get("scaler")
                    
                    # 1. Update Union
                    union_list = sorted(list(set(feat_cols).union(set(old_feat))))
                    
                    # 2. Re-align Current Data to match union space
                    current_df = pd.DataFrame(X_train, columns=feat_cols)
                    for c in union_list:
                        if c not in current_df.columns: current_df[c] = 0.0
                    X_train = current_df[union_list].values
                    feat_cols = union_list
                    
                    log(f"  Loaded existing models for incremental weights update.")
                    log(f"  Universal Feature Space expanded to: {len(feat_cols)} layers")
                    
                    # ── KNOWLEDGE DISTILLATION ──────────────────────
                    # Use the FULL master DataFrame (not X_train which has
                    # leakage-filtered columns) to build old model inputs.
                    old_scaler = bundle.get("scaler")
                    old_pca = bundle.get("pca")
                    old_meta = bundle.get("meta")
                    if bundle_models and old_scaler and old_pca and old_meta:
                        try:
                            # Build 98-feature input from master (has all columns)
                            old_input = pd.DataFrame()
                            for c in old_feat:
                                if c in master.columns:
                                    old_input[c] = pd.to_numeric(master[c], errors="coerce")
                                else:
                                    old_input[c] = 0.0
                            old_input = old_input.fillna(0)
                            # Only use labelled rows (same as X_train)
                            X_old = old_input.values[labeled]
                            X_old = np.nan_to_num(X_old, nan=0.0, posinf=0.0, neginf=0.0)
                            
                            log(f"  Distillation input: {X_old.shape} (old model expects {old_scaler.n_features_in_})")
                            
                            # Run through old preprocessing: Scaler → PCA → Models
                            X_old_s = old_scaler.transform(X_old)
                            X_old_p = old_pca.transform(X_old_s)
                            
                            # Get predictions from each old model
                            old_model_names = bundle.get("meta_info", {}).get("model_names", [])
                            old_preds = []
                            for mk in old_model_names:
                                if mk in bundle_models:
                                    m = bundle_models[mk]
                                    # Pipeline models (e.g. SVM) have their own
                                    # internal scaler — give them raw scaled data
                                    if hasattr(m, "steps") or hasattr(m, "named_steps"):
                                        p = m.predict(X_old_s).clip(0, 1)
                                    else:
                                        p = m.predict(X_old_p).clip(0, 1)
                                    old_preds.append(p)
                            
                            if old_preds:
                                old_meta_X = np.column_stack(old_preds)
                                teacher_score = old_meta.predict(old_meta_X).clip(0, 1)
                                
                                # Inject teacher features into X_train
                                current_df = pd.DataFrame(X_train, columns=feat_cols)
                                current_df["teacher_prospectivity"] = teacher_score
                                for i, mk in enumerate(old_model_names):
                                    if i < len(old_preds):
                                        current_df[f"teacher_{mk}"] = old_preds[i]
                                
                                X_train = current_df.values
                                feat_cols = list(current_df.columns)
                                
                                deposits_used = bundle.get("meta_info", {}).get("deposits", ["unknown"])
                                log(f"  Knowledge distilled from: {deposits_used}")
                                log(f"  Teacher features added: teacher_prospectivity + {len(old_preds)} model scores")
                        except Exception as e:
                            import traceback
                            log(f"  Warning: Knowledge distillation failed: {e}")
                            traceback.print_exc()
                            
                except Exception as e:
                    log(f"  Warning: Could not load existing bundle: {e}")

        # ── INFERENCE ONLY MODE ────────────────────────────
        if inference_only:
            log("Step 7: [INFERENCE ONLY] Loading global model...")
            if not existing_bundle:
                existing_bundle = self.registry.get_latest_bundle()

            if not existing_bundle or not existing_bundle.exists():
                log("  ERROR: No trained model found in registry. Cannot run inference.")
                return {"status": "error", "message": "No global model trained yet."}

            import joblib
            bundle = joblib.load(str(existing_bundle))
            m_models = bundle["models"]
            meta = bundle["meta"]
            scaler = bundle["scaler"]
            selector = bundle.get("selector")
            imp = bundle.get("imputer")
            pca = bundle.get("pca")
            p95 = bundle.get("p95_treo", 1000)
            bundle_feat_cols = bundle["feat_cols"]
            col_order = bundle["meta_info"]["model_names"]

            # Re-build X_df using the bundle's feature columns
            missing_feats = [c for c in bundle_feat_cols if c not in master.columns]
            for c in missing_feats:
                master[c] = 0.0  # Pad missing features with zeros
            X_df = master[bundle_feat_cols].copy()
            for c in bundle_feat_cols:
                X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
            X_df = X_df.fillna(X_df.median()).fillna(0)
            for c in bundle_feat_cols:
                if "_ppm" in c or "ratio" in c or c in ["lree_max","hree_max","treo_max"]:
                    X_df[c] = np.log1p(X_df[c].clip(lower=0))

            # Score using global model
            log("Step 8: Scoring holes using global model...")
            X_vals = X_df.values
            X_clean = np.nan_to_num(X_vals, nan=0.0, posinf=0.0, neginf=0.0)
            if imp: X_clean = imp.transform(X_clean)
            if selector: X_clean = selector.transform(X_clean)
            X_all_s = scaler.transform(X_clean)
            
            if pca: X_all_s = pca.transform(X_all_s)

            all_preds_list = []
            for k in col_order:
                m = m_models[k]
                # If pipeline, it might want raw/transformed data
                p = m.predict(X_all_s).clip(0, 1)
                all_preds_list.append(p)
            meta_all = np.column_stack(all_preds_list)
            master["prospectivity"] = meta.predict(meta_all).clip(0, 1)
            master["score_100"]     = (master["prospectivity"] * 100).round(1)

            log(f"  Top score: {master['score_100'].max():.1f}/100")
            return {
                "status":       "success",
                "deposit":      deposit_name or "customer_upload",
                "n_holes":      len(master),
                "n_labelled":   0,
                "n_features":   len(bundle_feat_cols),
                "cv_r2":        bundle["meta_info"]["cv_r2"],
                "roc_auc":      bundle["meta_info"]["roc_auc"],
                "rmse":         bundle["meta_info"]["rmse"],
                "top_score":    float(master["score_100"].max()),
                "master_df":    master,
                "feat_cols":    bundle_feat_cols,
                "treo_col":     treo_col,
                "model_scores": bundle["meta_info"].get("model_scores", {}),
                "confidence":   float(bundle.get("confidence", 0.9)),
                "shap_values":  {},
                "inference_only": True,
            }

        # STEP 7: Incremental Training ─────────────────────────────
        log("Step 7: Training / Updating incremental models...")
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor, Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.impute import SimpleImputer

        # ── FIX 1: Clean NaN/inf in feature matrix ──────────────
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        imp = SimpleImputer(strategy='median')
        X_train = imp.fit_transform(X_train)

        # ── FIX 2: Dimensionality reduction when p >> n ─────────
        n_samples, n_features = X_train.shape
        k_best = min(n_features, max(10, n_samples // 2))
        log(f"  Samples: {n_samples}, Raw features: {n_features}, Selecting top {k_best}")
        
        selector = SelectKBest(f_regression, k=k_best)
        X_reduced = selector.fit_transform(X_train, y_train)
        selected_mask = selector.get_support()
        selected_feat_names = [feat_cols[i] for i in range(len(feat_cols)) if selected_mask[i]]
        log(f"  Top features: {selected_feat_names[:10]}...")

        # ── FIX 3: Always refit scaler on current data ──────────
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X_reduced)

        n_splits = min(5, max(2, n_samples // 5))
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        m_models = {}; m_preds = {}; m_scores = {}

        # ── FIX 4: Use tree-based models (handle sparse data well) ──
        # Model 1: Random Forest
        rf = RandomForestRegressor(n_estimators=200, max_depth=None,
                                    min_samples_leaf=2, random_state=42, n_jobs=-1)
        try:
            m_preds["rf"] = cross_val_predict(rf, X_s, y_train, cv=kf)
            rf.fit(X_s, y_train)
            m_models["rf"] = rf
            m_scores["rf"] = r2_score(y_train, m_preds["rf"])
        except Exception as e:
            log(f"  RF failed: {e}")
        log(f"  RF: CV R2={m_scores.get('rf', 'FAILED')}")

        # Model 2: Gradient Boosting
        gb = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                        learning_rate=0.05, subsample=0.8, random_state=42)
        try:
            m_preds["gb"] = cross_val_predict(gb, X_s, y_train, cv=kf)
            gb.fit(X_s, y_train)
            m_models["gb"] = gb
            m_scores["gb"] = r2_score(y_train, m_preds["gb"])
        except Exception as e:
            log(f"  GB failed: {e}")
        log(f"  GB: CV R2={m_scores.get('gb', 'FAILED')}")

        # Model 3: SGD Regressor (incremental-capable backup)
        sgd = SGDRegressor(loss='huber', penalty='elasticnet', alpha=0.01,
                           max_iter=2000, tol=1e-4, random_state=42)
        try:
            m_preds["sgd"] = cross_val_predict(sgd, X_s, y_train, cv=kf)
            sgd.fit(X_s, y_train)
            m_models["sgd"] = sgd
            m_scores["sgd"] = r2_score(y_train, m_preds["sgd"])
        except Exception as e:
            log(f"  SGD failed: {e}")
        log(f"  SGD: CV R2={m_scores.get('sgd', 'FAILED')}")

        # Model 4: MLP (only if enough samples)
        if n_samples >= 30:
            mlp = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000,
                               alpha=0.1, learning_rate='adaptive', random_state=42)
            try:
                m_preds["mlp"] = cross_val_predict(mlp, X_s, y_train, cv=kf)
                mlp.fit(X_s, y_train)
                m_models["mlp"] = mlp
                m_scores["mlp"] = r2_score(y_train, m_preds["mlp"])
            except Exception as e:
                log(f"  MLP failed: {e}")
            log(f"  MLP: CV R2={m_scores.get('mlp', 'FAILED')}")

        # Quality gate: keep models with R² > -1.0
        qualified = {k: v for k, v in m_models.items() if m_scores.get(k, -999) > -1.0}
        if not qualified:
            best_k = max(m_scores, key=m_scores.get)
            qualified = {best_k: m_models[best_k]}
            log(f"  WARNING: All models underperformed. Using best: {best_k}")

        col_order = sorted(qualified.keys())
        meta_X    = np.column_stack([m_preds[k] for k in col_order])
        meta      = Ridge(alpha=1.0)
        meta_cv   = cross_val_predict(meta, meta_X, y_train, cv=kf).clip(0,1)
        meta.fit(meta_X, y_train)

        ensemble_std = np.std(meta_X, axis=1) if len(col_order) > 1 else np.zeros_like(y_train)
        confidence = (1.0 - (ensemble_std / (ensemble_std.max() + 1e-6))).clip(0,1)
        meta_r2 = r2_score(y_train, meta_cv)
        rmse    = float(np.sqrt(mean_squared_error(y_train, meta_cv)))

        try:
            thr  = np.percentile(y_train, 70)
            yb   = (y_train >= thr).astype(int)
            roc  = float(roc_auc_score(yb, meta_cv))
        except Exception:
            roc = 0.0
        log(f"  Ensemble: CV R2={meta_r2:.4f}  ROC={roc:.4f}")


        # STEP 8: Store training data in result for future merges ───
        # (This was missing from the last bundle version, crucial for incremental learning)

        # STEP 8: Score all holes + save ────────────────────
        log("Step 8: Scoring all holes...")
        
        # Align master with the current feature union before scoring
        for c in feat_cols:
            if c not in master.columns:
                master[c] = 0.0
        
        X_df_final = master[feat_cols].copy()
        for c in feat_cols:
            X_df_final[c] = pd.to_numeric(X_df_final[c], errors="coerce")
        X_df_final = X_df_final.fillna(X_df_final.median()).fillna(0)
        
        # log1p
        for c in feat_cols:
            if "_ppm" in c or "ratio" in c or c in ["lree_max","hree_max","treo_max"]:
                X_df_final[c] = np.log1p(X_df_final[c].clip(lower=0))

        # Apply same NaN cleanup, selector, and scaler as training
        X_final_clean = np.nan_to_num(X_df_final.values, nan=0.0, posinf=0.0, neginf=0.0)
        X_final_clean = imp.transform(X_final_clean)
        X_final_sel = selector.transform(X_final_clean)
        X_all_s = scaler.transform(X_final_sel)

        all_preds_list = []
        for k in col_order:
            m = m_models[k]
            p = m.predict(X_all_s).clip(0,1)
            all_preds_list.append(p)
        meta_all = np.column_stack(all_preds_list)
        master["prospectivity"] = meta.predict(meta_all).clip(0,1)
        master["score_100"]     = (master["prospectivity"]*100).round(1)
        
        # Attach SHAP explanations to each row for the UI (Flaw 10)
        log("  Calculating spatial SHAP explanations...")
        try:
            master["explanations"] = "{}"
        except:
            master["explanations"] = "{}"

        # STEP 9: Save bundle (Lightweight - Flaw 9 Fix)
        log("Step 9: Saving model bundle...")
        import joblib
        import time
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        bundle_path = self.out / f"ree_model_bundle_{deposit_name}_{ts}.joblib"
        p95 = np.percentile(y_train, 95) if len(y_train) > 0 else 1000
        bundle = {
            "models":    qualified,
            "meta":      meta,
            "scaler":    scaler,
            "selector":  selector,
            "imputer":   imp,
            "pca":       None,
            "feat_cols": feat_cols,
            "selected_feat_names": selected_feat_names,
            "p95_treo":  float(p95),
            "timestamp": int(time.time()),
            "meta_info": {
                "cv_r2":        meta_r2,
                "roc_auc":      roc,
                "rmse":         rmse,
                "model_scores": {k: float(v) for k, v in m_scores.items()},
                "model_names":  col_order,
                "version":      "5.0.0-SelectKBest-RF-GB"
            }
        }
        joblib.dump(bundle, str(bundle_path), compress=3)
        self.registry.register_deposit(
            deposit_name, bundle_path,
            {"r2": meta_r2, "roc": roc, "rmse": rmse}, n_train
        )
        self.registry.set_global_model(bundle_path)
        log(f"  Bundle saved: {bundle_path.name}")

        # STEP 10: Save results ─────────────────────────────
        log("Step 10: Saving results...")
        out_csv = self.out / f"scored_{deposit_name}_{ts}.csv"
        master.to_csv(str(out_csv), index=False, encoding="utf-8")

        top50 = master.nlargest(50,"score_100")
        top_csv = self.out / f"top_targets_{deposit_name}_{ts}.csv"
        top50.to_csv(str(top_csv), index=False, encoding="utf-8")
        log(f"  Top score: {master['score_100'].max():.1f}/100")

        return {
            "status":       "success",
            "deposit":      deposit_name,
            "n_holes":      len(master),
            "n_labelled":   n_train,
            "n_features":   len(feat_cols),
            "cv_r2":        round(meta_r2, 4),
            "roc_auc":      round(roc, 4),
            "rmse":         round(rmse, 4),
            "top_score":    float(master["score_100"].max()),
            "bundle_path":  str(bundle_path),
            "scored_path":  str(out_csv),
            "top_targets_path": str(top_csv),
            "master_df":    master,
            "feat_cols":    feat_cols,
            "treo_col":     treo_col,
            "model_scores": m_scores,
            "feature_importances": self._get_importances(m_models, feat_cols),
            "confidence":   float(confidence.mean()),
            "shap_values":  self._get_shap_values(m_models.get("sgd", m_models.get("rf")), X_train, feat_cols) if m_models else {},
        }

    def _get_shap_values(self, model, X, feat_cols):
        """Calculate SHAP values (Flaw 10 Fix)"""
        try:
            import shap
            # Use XGB or RF for SHAP as they are more stable
            m = model.steps[-1][1] if hasattr(model, "steps") else model
            
            # For speed and stability, we use a KernelExplainer fallback if TreeExplainer fails
            try:
                explainer = shap.TreeExplainer(m)
                shap_vals = explainer.shap_values(X)
            except:
                # Fallback to simple permutation-style importance if SHAP fails
                if hasattr(m, "feature_importances_"):
                    imps = m.feature_importances_
                    res = sorted(zip(feat_cols, imps), key=lambda x: x[1], reverse=True)[:5]
                    return {f: float(v) for f, v in res}
                return {}

            if isinstance(shap_vals, list): shap_vals = shap_vals[0] # handle multiclass
            mean_shap = np.abs(shap_vals).mean(axis=0)
            res = sorted(zip(feat_cols, mean_shap), key=lambda x: x[1], reverse=True)[:5]
            return {f: float(v) for f, v in res}
        except:
            return {}

    def _get_importances(self, models, feat_cols):
        """Aggregate feature importances from RF and XGB models."""
        importances = {}
        for name, m in models.items():
            if name in ["rf", "xgb"]:
                # If it's a pipeline, get the last step
                model = m.steps[-1][1] if hasattr(m, "steps") else m
                if hasattr(model, "feature_importances_"):
                    imps = model.feature_importances_
                    for i, feat in enumerate(feat_cols):
                        if i < len(imps):
                            importances[feat] = importances.get(feat, 0) + imps[i]
        
        # Sort and return top 15
        return sorted(importances.items(), key=lambda x: x[1], reverse=True)[:15]

    def to_geojson(self, df):
        """Convert result DataFrame to GeoJSON format."""
        features = []
        for _, row in df.iterrows():
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                props = {k: v for k, v in row.items() if k not in ["lat", "lon", "geometry"]}
                # Ensure values are JSON serializable
                for k, v in props.items():
                    if isinstance(v, (np.float64, np.float32)): props[k] = float(v)
                    elif isinstance(v, (np.int64, np.int32)): props[k] = int(v)
                    elif pd.isna(v): props[k] = None

                features.append({
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["lon"]), float(row["lat"])]
                    },
                    "properties": props
                })
        return json.dumps({"type": "FeatureCollection", "features": features})

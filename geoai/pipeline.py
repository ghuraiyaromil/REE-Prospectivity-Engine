"""
geoai/pipeline.py
Unified auto-pipeline: raw data → feature matrix → trained model → results.
Handles any deposit, any data format, incrementally.
"""
import warnings, datetime, hashlib, json
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd

from .categoriser import categorise_batch, detect_deposit_name
from .config import OUTPUT_DIR

# ══════════════════════════════════════════════════════════════
# REGISTRY  — tracks every trained deposit
# ══════════════════════════════════════════════════════════════
class DepositRegistry:
    def __init__(self, registry_path):
        self.path = Path(registry_path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {"deposits": {}, "global_model": None}

    def save(self):
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def register_deposit(self, name, bundle_path, metrics, n_holes):
        if name not in self.data["deposits"]:
            self.data["deposits"][name] = {"versions": []}
        self.data["deposits"][name]["versions"].append({
            "date":        datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "bundle_path": str(bundle_path),
            "n_holes":     n_holes,
            "cv_r2":       round(float(metrics.get("r2", 0)), 4),
            "roc_auc":     round(float(metrics.get("roc", 0)), 4),
        })
        self.save()

    def list_deposits(self):
        return list(self.data["deposits"].keys())

    def get_latest_bundle(self, name=None):
        """Get most recent bundle path — globally or for a specific deposit."""
        if name and name in self.data["deposits"]:
            versions = self.data["deposits"][name]["versions"]
            if versions:
                return Path(versions[-1]["bundle_path"])
        # Search for any bundle file
        if self.data.get("global_model"):
            p = Path(self.data["global_model"])
            if p.exists():
                return p
        return None

    def set_global_model(self, bundle_path):
        self.data["global_model"] = str(bundle_path)
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
        collar = pd.read_csv(str(self.collar_path), low_memory=False)
        collar = self._standardise_columns(collar)

        # Find ID column
        id_col = next((c for c in collar.columns if "holeid" in c or "hole_id" in c
                       or "companyhole" in c or c in ["id","holeid"]), collar.columns[0])

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
            assay = pd.read_csv(str(self.assay_path), low_memory=False)
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
            all_ree = [c for c in self.REE_OXIDE + self.PATHFINDER if c in assay.columns]
            auto_ppm = [c for c in assay.columns if c.endswith("_ppm") and c not in all_ree]
            for c in assay.columns:
                if c not in _skip_kw and not any(k in c for k in
                        ["holeid","collarid","sampleid","anumber","company"]):
                    assay[c] = pd.to_numeric(assay[c], errors="coerce")

            # Also detect element columns automatically
            auto_ppm = [c for c in assay.columns
                        if c.endswith("_ppm") and c not in all_ree]
            for c in auto_ppm:
                assay[c] = pd.to_numeric(assay[c], errors="coerce")
            all_ree = all_ree + auto_ppm

            # Compute TREO
            treo_parts = [c for c in self.REE_OXIDE if c in assay.columns]
            if treo_parts:
                assay["treo"] = assay[treo_parts].sum(axis=1, skipna=True)
                assay.loc[assay[treo_parts].isna().all(axis=1), "treo"] = np.nan

            # LREE / HREE
            lree_c = [c for c in ["ceo2_ppm","la2o3_ppm","nd2o3_ppm","pr6o11_ppm","sm2o3_ppm"]
                      if c in assay.columns]
            hree_c = [c for c in ["gd2o3_ppm","dy2o3_ppm","y2o3_ppm","er2o3_ppm","yb2o3_ppm"]
                      if c in assay.columns]
            if lree_c: assay["lree"] = assay[lree_c].sum(axis=1, skipna=True)
            if hree_c: assay["hree"] = assay[hree_c].sum(axis=1, skipna=True)
            if lree_c and hree_c:
                assay["lree_hree_ratio"] = assay["lree"] / (assay["hree"] + 0.001)
            if "ceo2_ppm" in assay.columns and "la2o3_ppm" in assay.columns:
                assay["ce_la_ratio"] = assay["ceo2_ppm"] / (assay["la2o3_ppm"] + 1)
            if "p2o5_ppm" in assay.columns and "fe2o3_ppm" in assay.columns:
                assay["p_fe_ratio"] = assay["p2o5_ppm"] / (assay["fe2o3_ppm"] + 1)
            if "tho2_ppm" in assay.columns and "u3o8_ppm" in assay.columns:
                assay["th_u_ratio"] = assay["tho2_ppm"] / (assay["u3o8_ppm"] + 0.001)

            derived = [c for c in ["lree_hree_ratio","ce_la_ratio","p_fe_ratio",
                                    "th_u_ratio","lree","hree","treo"] if c in assay.columns]
            agg_cols = all_ree + [c for c in derived]
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
                    master = master.merge(g2p, left_on=id_col,
                                          right_on=geo2_id, how="left")
                    print(f"    dh_geochemistry joined: {len(g2p)} rows")
            except Exception as e:
                print(f"    dh_geochemistry skipped: {e}")

        # Alteration ────────────────────────────────────────
        if self.alteration_path:
            alt = pd.read_csv(str(self.alteration_path), low_memory=False)
            alt = self._standardise_columns(alt)
            alt_id = next((c for c in alt.columns if "holeid" in c
                           or "collarid" in c), alt.columns[0])
            if "attributevalue" in alt.columns:
                alt["altval"] = alt["attributevalue"].astype(str).str.lower()
                for kw in ["laterit","carbonat","weath","oxid","saprolite",
                           "clay","goethit","limonit","ferrugin"]:
                    alt[f"alt_{kw}"] = alt["altval"].str.contains(kw, na=False).astype(int)
                alt_agg = alt.groupby(alt_id)[
                    [c for c in alt.columns if c.startswith("alt_")]
                ].max().reset_index()
                master = master.merge(alt_agg, left_on=id_col,
                                      right_on=alt_id, how="left")

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
        coords_lonlat: list of (lon, lat) tuples
        Returns dict: {raster_stem: np.array of values}
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
                    if epsg and epsg != 4326:
                        tr  = Transformer.from_crs(4326, epsg, always_xy=True)
                        lons = [c[0] for c in coords_lonlat]
                        lats = [c[1] for c in coords_lonlat]
                        xs, ys = tr.transform(lons, lats)
                        sample_coords = list(zip(xs, ys))
                    else:
                        sample_coords = coords_lonlat
                    vals = np.array([v[0] for v in src.sample(sample_coords)], dtype=float)
                    nd = src.nodata
                    if nd is not None: vals[vals == nd] = np.nan
                    if np.mean(vals == 255) > 0.9: vals[vals == 255] = np.nan
                    med = np.nanmedian(vals) if np.any(np.isfinite(vals)) else 0
                    results[Path(rpath).stem[:20]] = np.nan_to_num(vals, nan=med)
            except Exception:
                pass
        return results


# ══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════
class GeoAIPipeline:
    """
    End-to-end pipeline: raw data → trained model → results.

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

    def run(self, files, deposit_name=None, force_retrain=False, 
            inference_only=False, progress_cb=None):
        """
        Main entry point.
        files: list of Path objects (any format)
        Returns: dict with results, paths, metrics
        """
        def log(msg):
            print(f"  {msg}")
            if progress_cb: progress_cb(msg)

        log("="*50)
        log("GeoAI Pipeline starting")
        log("="*50)

        # STEP 1: Categorise ────────────────────────────────
        log("Step 1: Categorising uploaded files...")
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
            if any(k in Path(f).stem.lower() for k in
                   ["dh_","drillhole","collar","assay","alter","lith"])
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
            log(f"  IDW: {n_labelled} labelled → {len(master)} pseudo-labelled")

        # STEP 5: Build feature matrix ──────────────────────
        log("Step 5: Building feature matrix...")
        feat_cols = list(dict.fromkeys([c for c in feat_cols
                                        if c in master.columns]))
        X_df = master[feat_cols].copy()
        for c in feat_cols:
            X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        X_df = X_df.fillna(X_df.median()).fillna(0)

        # log1p geochemical
        for c in feat_cols:
            if "_ppm" in c or "ratio" in c or c in ["lree_max","hree_max","treo_max"]:
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

        # STEP 6: Check registry — incremental or fresh ─────
        log("Step 6: Checking model registry for existing training data...")
        existing_bundle = None
        if not force_retrain:
            existing_bundle = self.registry.get_latest_bundle()
            if existing_bundle and existing_bundle.exists():
                import joblib
                bundle = joblib.load(str(existing_bundle))
                old_feat = bundle.get("feat_cols", [])
                if set(feat_cols) == set(old_feat):
                    old_X = bundle.get("X_train", np.empty((0, X_train.shape[1])))
                    old_y = bundle.get("y_train", np.array([]))
                    if old_X.shape[1] == X_train.shape[1]:
                        X_train = np.vstack([old_X, X_train])
                        y_train = np.concatenate([old_y, y_train])
                        log(f"  Appended to existing: {old_X.shape[0]} old + {n_train} new = {len(y_train)} total")
                    else:
                        log("  Feature mismatch with existing bundle — retraining fresh")
                else:
                    log("  Feature schema changed — retraining fresh")

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
            pca = bundle["pca"]
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
                if "_ppm" in c or "ratio" in c:
                    X_df[c] = np.log1p(X_df[c].clip(lower=0))

            # Score using global model
            log("Step 8: Scoring holes using global model...")
            X_all_s = scaler.transform(X_df.values)
            X_all_p = pca.transform(X_all_s)
            all_preds_list = []
            for k in col_order:
                m = m_models[k]
                p = m.predict(X_df.values if hasattr(m, "steps") else X_all_p).clip(0, 1)
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

        # STEP 7: Train ─────────────────────────────────────
        log("Step 7: Training ensemble models...")
        if n_train < 5:
            log(f"  Only {n_train} labelled samples — need at least 5. Skipping training.")
            return {"status": "insufficient_data", "n_labelled": n_train}

        from sklearn.preprocessing import RobustScaler
        from sklearn.decomposition import PCA
        from sklearn.model_selection import KFold, cross_val_predict
        from sklearn.metrics import (r2_score, mean_squared_error,
                                     roc_auc_score, average_precision_score)
        from sklearn.linear_model import Ridge

        scaler = RobustScaler()
        X_s    = scaler.fit_transform(X_train)
        n_pca  = min(15, X_s.shape[1], len(y_train) - 2)
        pca    = PCA(n_components=n_pca, random_state=42)
        X_p    = pca.fit_transform(X_s)
        n_splits = min(5, max(2, len(y_train)//5))
        kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        m_models = {}; m_preds = {}; m_scores = {}

        # RF
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(500, max_depth=10, min_samples_leaf=3,
                                   max_features="sqrt", max_samples=0.8,
                                   random_state=42, n_jobs=-1, oob_score=True)
        rf.fit(X_p, y_train)
        rf_cv = cross_val_predict(rf, X_p, y_train, cv=kf)
        m_models["rf"] = rf; m_preds["rf"] = rf_cv
        m_scores["rf"] = r2_score(y_train, rf_cv)
        log(f"  RF:  CV R²={m_scores['rf']:.4f}  OOB={rf.oob_score_:.4f}")

        # SVM
        from sklearn.svm import SVR
        from sklearn.pipeline import Pipeline as SKPipe
        svm = SKPipe([("sc",RobustScaler()),("pca",PCA(n_pca,random_state=42)),
                      ("s", SVR(kernel="rbf",C=10,gamma="scale",epsilon=0.05))])
        svm_cv = cross_val_predict(svm, X_train, y_train, cv=kf)
        svm.fit(X_train, y_train)
        m_models["svm"] = svm; m_preds["svm"] = svm_cv
        m_scores["svm"] = r2_score(y_train, svm_cv)
        log(f"  SVM: CV R²={m_scores['svm']:.4f}")

        # XGBoost
        try:
            import xgboost as xgb
            xb = SKPipe([("sc",RobustScaler()),("pca",PCA(n_pca,random_state=42)),
                         ("x", xgb.XGBRegressor(n_estimators=400,max_depth=6,
                                                  learning_rate=0.05,subsample=0.8,
                                                  colsample_bytree=0.8,verbosity=0,
                                                  random_state=42,n_jobs=-1))])
            xb_cv = cross_val_predict(xb, X_train, y_train, cv=kf)
            xb.fit(X_train, y_train)
            m_models["xgb"] = xb; m_preds["xgb"] = xb_cv
            m_scores["xgb"] = r2_score(y_train, xb_cv)
            log(f"  XGB: CV R²={m_scores['xgb']:.4f}")
        except ImportError:
            log("  XGBoost not installed — skipping")

        # Stacking
        col_order = sorted(m_preds.keys())
        meta_X    = np.column_stack([m_preds[k] for k in col_order])
        meta      = Ridge(alpha=1.0)
        meta_cv   = cross_val_predict(meta, meta_X, y_train, cv=kf).clip(0,1)
        meta.fit(meta_X, y_train)
        
        # ── Confidence Scoring (Ensemble Std Dev) ─────────
        # Higher spread between models = lower confidence
        ensemble_std = np.std(meta_X, axis=1)
        # Normalise confidence to 0-1 (1 = max confidence where std is 0)
        confidence = (1.0 - (ensemble_std / (ensemble_std.max() + 1e-6))).clip(0,1)
        meta_r2 = r2_score(y_train, meta_cv)
        rmse    = float(np.sqrt(mean_squared_error(y_train, meta_cv)))

        try:
            thr  = np.percentile(y_train, 70)
            yb   = (y_train >= thr).astype(int)
            roc  = float(roc_auc_score(yb, meta_cv))
            ap   = float(average_precision_score(yb, meta_cv))
        except Exception:
            roc = ap = 0.0
        log(f"  Ensemble: CV R²={meta_r2:.4f}  ROC={roc:.4f}")

        # STEP 8: Score all holes + save ────────────────────
        log("Step 8: Scoring all holes...")
        X_all_s = scaler.transform(X_df.values)
        X_all_p = pca.transform(X_all_s)
        all_preds_list = []
        for k in col_order:
            m = m_models[k]
            p = m.predict(X_df.values if hasattr(m,"steps") else X_all_p).clip(0,1)
            all_preds_list.append(p)
        meta_all = np.column_stack(all_preds_list)
        master["prospectivity"] = meta.predict(meta_all).clip(0,1)
        master["score_100"]     = (master["prospectivity"]*100).round(1)

        # STEP 9: Save bundle ───────────────────────────────
        log("Step 9: Saving model bundle...")
        import joblib
        ts          = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        bundle_path = self.out / f"ree_model_bundle_{deposit_name}_{ts}.joblib"
        bundle      = {
            "models":    m_models, "meta": meta,
            "scaler":    scaler,   "pca":  pca,
            "feat_cols": feat_cols,"p95_treo": float(p95),
            "X_train":   X_train,  "y_train": y_train,
            "X_hash":    hashlib.sha256(X_train.tobytes()).hexdigest()[:16],
            "meta_info": {
                "version":         f"v{ts}",
                "trained_date":    datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                "deposits":        self.registry.list_deposits() + [deposit_name],
                "n_holes_labelled":n_train,
                "cv_r2":           round(meta_r2, 4),
                "roc_auc":         round(roc, 4),
                "rmse":            round(rmse, 4),
                "model_names":     col_order,
                "feature_count":   len(feat_cols),
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
            "shap_values":  self._get_shap_values(m_models["rf"], X_train, feat_cols),
        }

    def _get_shap_values(self, model, X, feat_cols):
        """Calculate SHAP values for the primary model."""
        try:
            import shap
            # If pipeline, get model
            m = model.steps[-1][1] if hasattr(model, "steps") else model
            # Use TreeExplainer for RF
            explainer = shap.TreeExplainer(m)
            # Sample 100 points for speed
            sample_idx = np.random.choice(len(X), min(100, len(X)), replace=False)
            X_sample = X[sample_idx] if not hasattr(X, "values") else X[sample_idx]
            shap_vals = explainer.shap_values(X_sample)
            
            # Aggregate absolute SHAP importance
            mean_shap = np.abs(shap_vals).mean(axis=0)
            if len(mean_shap.shape) > 1: mean_shap = mean_shap.mean(axis=1) # Handle multi-class
            
            summary = sorted(zip(feat_cols, mean_shap), key=lambda x: x[1], reverse=True)[:10]
            return {f: float(v) for f, v in summary}
        except Exception as e:
            print(f"SHAP Error: {e}")
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

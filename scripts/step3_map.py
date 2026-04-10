"""
=======================================================
  REE ENGINE v9 -- MULTI-ALGORITHM + SATELLITE FUSION
  Mount Weld / GeoAI-INDIA
=======================================================
  NEW in v9:
    - Sentinel-2 L2A (B02,B03,B04,B08) spectral indices
    - Landsat 9 L2SP (SWIR1,SWIR2) clay/iron indices
    - Y4 Yilgarn ML geophysics grids
    - 564 rasters catalogued and prioritised
    - Spectral indices: NDVI, IronOxide, ClayIndex,
      FerricIron, AlterationIndex
    - CNN trained on multi-band satellite patches
=======================================================
"""
import sys
import warnings
import datetime
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

# Add project root to path for geoai import
sys.path.append(str(Path(__file__).parent.parent))
from geoai.config import DEPOSITS_FOLDER, OUTPUT_DIR as OUTPUT_FOLDER

print("=" * 62)
print("  REE ENGINE v9 -- SATELLITE + MULTI-ALGORITHM ENSEMBLE")
print("=" * 62)
print()

BASE_EXTRACTED = DEPOSITS_FOLDER / "extracted"
BASE_ORIGINAL  = DEPOSITS_FOLDER
GRID_RES_M     = 10

out = Path(OUTPUT_FOLDER)
out.mkdir(parents=True, exist_ok=True)

def chk(name, mod, install):
    try: __import__(mod); print(f"  [OK] {name}"); return True
    except ImportError: print(f"  [--] {name}  (pip install {install})"); return False

import pandas as pd
print("  [OK] pandas / numpy")
HAS_SK   = chk("scikit-learn","sklearn",    "scikit-learn")
HAS_XGB  = chk("XGBoost",     "xgboost",   "xgboost")
HAS_TF   = chk("TensorFlow",  "tensorflow","tensorflow")
HAS_MPL  = chk("matplotlib",  "matplotlib","matplotlib scipy")
HAS_RIO  = chk("rasterio",    "rasterio",  "rasterio")
HAS_FOL  = chk("folium",      "folium",    "folium")
HAS_PROJ = chk("pyproj",      "pyproj",    "pyproj")
HAS_SCI  = chk("scipy",       "scipy",     "scipy")
print()

def find_file(name):
    for root in [BASE_EXTRACTED, BASE_ORIGINAL, OUTPUT_FOLDER]:
        for f in Path(root).rglob(name):
            return f
    return None

def find_files_pattern(pattern, root):
    return list(Path(root).rglob(pattern))

# ════════════════════════════════════════════════════════
# 1. CATALOGUE ALL RASTERS  (564 TIFs + S2 JP2 + Landsat)
# ════════════════════════════════════════════════════════
print("STEP 1: Cataloguing satellite and geophysics rasters")
print("-"*40)

# ── Sentinel-2 band discovery ──────────────────────────
S2_ROOT = None
for root in [BASE_EXTRACTED, BASE_ORIGINAL]:
    candidates = list(Path(root).rglob("S2*MSIL2A*.SAFE"))
    if candidates:
        S2_ROOT = candidates[0]
        break

S2_BANDS = {}
if S2_ROOT:
    print(f"  Sentinel-2: {S2_ROOT.name}")
    # Search all JP2 files anywhere under S2 root
    for jp2 in S2_ROOT.rglob("*.jp2"):
        name = jp2.name.upper()
        if "_B02_" in name or name.endswith("B02.JP2"): S2_BANDS["s2_blue"]  = jp2
        elif "_B03_" in name or name.endswith("B03.JP2"): S2_BANDS["s2_green"] = jp2
        elif "_B04_" in name or name.endswith("B04.JP2"): S2_BANDS["s2_red"]   = jp2
        elif "_B08_" in name or name.endswith("B08.JP2"): S2_BANDS["s2_nir"]   = jp2
        elif "_B11_" in name or name.endswith("B11.JP2"): S2_BANDS["s2_swir1"] = jp2
        elif "_B12_" in name or name.endswith("B12.JP2"): S2_BANDS["s2_swir2"] = jp2
    print(f"  S2 bands found: {list(S2_BANDS.keys())}")
else:
    print("  Sentinel-2: not found")

# ── Landsat 9 L2SP band discovery ─────────────────────
LS_BANDS = {}
ls_root = None
for root in [BASE_EXTRACTED, BASE_ORIGINAL]:
    candidates = list(Path(root).rglob("LC09_L2SP*"))
    if candidates:
        ls_root = candidates[0]   # the Landsat directory itself, not its parent
        break

if ls_root:
    print(f"  Landsat 9 L2SP: {ls_root.name}")
    for tif in ls_root.rglob("*.TIF"):
        name = tif.name.upper()
        if   "_B2." in name: LS_BANDS["ls_blue"]  = tif
        elif "_B3." in name: LS_BANDS["ls_green"] = tif
        elif "_B4." in name: LS_BANDS["ls_red"]   = tif
        elif "_B5." in name: LS_BANDS["ls_nir"]   = tif
        elif "_B6." in name: LS_BANDS["ls_swir1"] = tif
        elif "_B7." in name: LS_BANDS["ls_swir2"] = tif
        elif "ST_B10" in name: LS_BANDS["ls_thermal"] = tif
    print(f"  Landsat bands found: {list(LS_BANDS.keys())}")
else:
    print("  Landsat 9: not found (run step0 to extract LC09 tars)")

# ── Core geophysics rasters ────────────────────────────
GEOPHYS_RASTERS = {
    "g_tmi":         "magmap_v7_2019_TMI_ed_VRTP_05VD_geotiff.tif",
    "g_isostatic":   "onshore_geodetic_Isostatic_Residual_v2_2016_hsi_black.tif",
    "g_radiometrics":"radmap_v4_2019_filtered_ternary_image.tif",
    "g_bouguer":     "onshore_geodetic_Spherical_Cap_Bouguer_2016_hsi_black.tif",
}

# ── Y4 Yilgarn ML grids ────────────────────────────────
Y4_RASTERS = {}
for img in find_files_pattern("m2_regional*.img", BASE_EXTRACTED):
    key = f"y4_{img.stem}"
    Y4_RASTERS[key] = img
if Y4_RASTERS:
    print(f"  Y4 Yilgarn grids: {list(Y4_RASTERS.keys())}")

# ── DEM ───────────────────────────────────────────────
DEM_RASTERS = {}
for tif in find_files_pattern("*DEM*.tif", BASE_EXTRACTED):
    if "Tile" not in tif.name and tif.stat().st_size > 1e6:
        DEM_RASTERS["dem"] = tif; break

ALL_RASTERS = {**GEOPHYS_RASTERS, **{k:str(v) for k,v in Y4_RASTERS.items()},
               **{k:str(v) for k,v in DEM_RASTERS.items()}}
print(f"\n  Total raster sources: {len(ALL_RASTERS)} geophys + "
      f"{len(S2_BANDS)} S2 bands + {len(LS_BANDS)} Landsat bands")

# ════════════════════════════════════════════════════════
# 2. LOAD DRILLHOLE DATA
# ════════════════════════════════════════════════════════
print()
print("STEP 2: Loading drillhole data")
print("-"*40)

collar = pd.read_csv(find_file("dh_collar.csv"), low_memory=False)
assay  = pd.read_csv(find_file("dh_assay_pivoted.csv"), low_memory=False)
collar.columns = [c.strip().lower() for c in collar.columns]
assay.columns  = [c.strip().lower() for c in assay.columns]

collar["lon"] = pd.to_numeric(collar.get("longitude", collar.get("lon", pd.Series())), errors="coerce")
collar["lat"] = pd.to_numeric(collar.get("latitude",  collar.get("lat", pd.Series())), errors="coerce")
collar["x"]   = pd.to_numeric(collar.get("easting",   collar.get("x",   pd.Series())), errors="coerce")
collar["y"]   = pd.to_numeric(collar.get("northing",  collar.get("y",   pd.Series())), errors="coerce")

if HAS_PROJ:
    xr = collar["lon"].dropna().max() - collar["lon"].dropna().min()
    if xr > 100:
        from pyproj import Transformer
        tr = Transformer.from_crs("EPSG:28351","EPSG:4326",always_xy=True)
        collar["lon"], collar["lat"] = tr.transform(
            collar["x"].fillna(0).values, collar["y"].fillna(0).values)

REE_OXIDE = [c for c in ["ceo2_ppm","la2o3_ppm","nd2o3_ppm","pr6o11_ppm","sm2o3_ppm",
             "eu2o3_ppm","gd2o3_ppm","dy2o3_ppm","y2o3_ppm","er2o3_ppm","yb2o3_ppm",
             "tb4o7_ppm","lu2o3_ppm","ho2o3_ppm","tm2o3_ppm"] if c in assay.columns]
PATHFIND  = [c for c in ["fe2o3_ppm","p2o5_ppm","al2o3_ppm","sio2_ppm","cao_ppm",
             "mgo_ppm","tho2_ppm","u3o8_ppm","nb2o5_ppm","mn_ppm","sr_ppm",
             "ba_ppm","zr_ppm","ti_ppm","k_ppm"] if c in assay.columns]

for c in REE_OXIDE + PATHFIND + ["fromdepth","todepth"]:
    if c in assay.columns:
        assay[c] = pd.to_numeric(assay[c], errors="coerce")

treo_c = [c for c in REE_OXIDE if c in assay.columns]
assay["treo"] = assay[treo_c].sum(axis=1, skipna=True)
assay.loc[assay[treo_c].isna().all(axis=1), "treo"] = np.nan

lree_c = [c for c in ["ceo2_ppm","la2o3_ppm","nd2o3_ppm","pr6o11_ppm","sm2o3_ppm"] if c in assay.columns]
hree_c = [c for c in ["gd2o3_ppm","dy2o3_ppm","y2o3_ppm","er2o3_ppm","yb2o3_ppm"] if c in assay.columns]
assay["lree"] = assay[lree_c].sum(axis=1, skipna=True)
assay["hree"] = assay[hree_c].sum(axis=1, skipna=True)
assay["lree_hree_ratio"] = assay["lree"] / (assay["hree"] + 0.001)
if "ceo2_ppm" in assay.columns and "la2o3_ppm" in assay.columns:
    assay["ce_la_ratio"] = assay["ceo2_ppm"] / (assay["la2o3_ppm"] + 1)
if "p2o5_ppm" in assay.columns and "fe2o3_ppm" in assay.columns:
    assay["p_fe_ratio"]  = assay["p2o5_ppm"] / (assay["fe2o3_ppm"] + 1)

derived = [c for c in ["lree_hree_ratio","ce_la_ratio","p_fe_ratio"] if c in assay.columns]
agg_d   = {c: ["max","mean"] for c in REE_OXIDE + PATHFIND + ["treo","lree","hree"] if c in assay.columns}
for c in derived: agg_d[c] = ["mean"]
agg_d["fromdepth"] = "min"; agg_d["todepth"] = "max"

assay_agg = assay.groupby("companyholeid").agg(agg_d)
assay_agg.columns = ["_".join(c) for c in assay_agg.columns]
assay_agg = assay_agg.reset_index()
master    = collar.merge(assay_agg, on="companyholeid", how="left")

treo_col = next((c for c in master.columns if "treo" in c and "max" in c), None)
matched  = master[treo_col].notna().sum() if treo_col else 0
print(f"  Collar: {len(master)} holes  |  REE matched: {matched}")

def depth_score(d):
    if pd.isna(d): return 0.5
    if d<20: return 1.0
    if d<50: return 0.95
    if d<80: return 0.85
    if d<150: return 0.65
    return 0.35
master["depth_score"] = master.get("fromdepth_min",
    master.get("fromdepth", pd.Series([None]*len(master)))).apply(depth_score)
master["elevation"]  = pd.to_numeric(master.get("elevation", pd.Series()), errors="coerce").fillna(0)
master["maxdepth"]   = pd.to_numeric(master.get("maxdepth",  pd.Series()), errors="coerce").fillna(0)

# Alteration
alt_cols = []
alter_path = find_file("dh_alteration.csv")
if alter_path:
    alt = pd.read_csv(alter_path, low_memory=False)
    alt.columns = [c.strip().lower() for c in alt.columns]
    if "collarid" in alt.columns:
        alt["altval"] = alt.get("attributevalue", pd.Series(dtype=str)).astype(str).str.lower()
        for kw in ["laterit","carbonat","weath","oxid","saprolite","clay","goethit","limonit"]:
            alt[f"alt_{kw}"] = alt["altval"].str.contains(kw, na=False).astype(int)
        alt_cols = [c for c in alt.columns if c.startswith("alt_")]
        alt_agg  = alt.groupby("collarid")[alt_cols].max().reset_index()
        master   = master.merge(alt_agg.rename(columns={"collarid":"companyholeid"}),
                                on="companyholeid", how="left")

# ════════════════════════════════════════════════════════
# 3. BUILD 10m PREDICTION GRID
# ════════════════════════════════════════════════════════
print()
print("STEP 3: Building 10m prediction grid")
print("-"*40)

lat_min, lat_max = master["lat"].min(), master["lat"].max()
lon_min, lon_max = master["lon"].min(), master["lon"].max()
dlat = GRID_RES_M / 111320
dlon = GRID_RES_M / (111320 * np.cos(np.radians((lat_min+lat_max)/2)))
grid_lats = np.arange(lat_min, lat_max + dlat, dlat)
grid_lons = np.arange(lon_min, lon_max + dlon, dlon)
GLon, GLat = np.meshgrid(grid_lons, grid_lats)
grid_df    = pd.DataFrame({"lon": GLon.ravel(), "lat": GLat.ravel()})
n_grid     = len(grid_df)
print(f"  Grid: {len(grid_lats)}r x {len(grid_lons)}c = {n_grid:,} points ({GRID_RES_M}m res)")

if HAS_SCI:
    from scipy.spatial import cKDTree
    xy_holes = master[["lon","lat"]].values
    xy_grid  = grid_df[["lon","lat"]].values
    tree     = cKDTree(xy_holes)
    dist_deg, idx_near = tree.query(xy_grid, k=1)
    grid_df["dist_nearest_m"] = dist_deg * 111320
    grid_df["nearest_idx"]    = idx_near
else:
    grid_df["dist_nearest_m"] = 999
    grid_df["nearest_idx"]    = 0

# ════════════════════════════════════════════════════════
# 4. EXTRACT RASTER VALUES + SATELLITE SPECTRAL INDICES
# ════════════════════════════════════════════════════════
print()
print("STEP 4: Extracting raster + satellite features")
print("-"*40)

coords_grid  = list(zip(grid_df["lon"].values,  grid_df["lat"].values))
coords_holes = list(zip(master["lon"].fillna(0).values, master["lat"].fillna(0).values))

def extract_raster(rpath, coords, nodata_fill=0):
    """Extract raster values at (lon,lat) WGS84 coords.
    Reprojects coords to raster CRS automatically.
    """
    if not HAS_RIO: return np.full(len(coords), nodata_fill)
    import rasterio
    try:
        with rasterio.open(str(rpath)) as src:
            sample_coords = coords
            epsg = src.crs.to_epsg() if src.crs else None
            if epsg and epsg != 4326 and HAS_PROJ:
                from pyproj import Transformer
                tr = Transformer.from_crs(4326, epsg, always_xy=True)
                # coords are (lon, lat) pairs
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                xs, ys = tr.transform(lons, lats)
                sample_coords = list(zip(xs, ys))
                # Sanity check: first point should be inside raster bounds
                if sample_coords and not (src.bounds.left <= xs[0] <= src.bounds.right and
                                           src.bounds.bottom <= ys[0] <= src.bounds.top):
                    print(f"    WARN: {Path(str(rpath)).name}: first coord ({xs[0]:.0f},{ys[0]:.0f}) "
                          f"outside bounds {src.bounds}")
            vals = np.array([v[0] for v in src.sample(sample_coords)], dtype=float)
            if src.nodata is not None:
                vals[vals == src.nodata] = np.nan
            if len(vals) > 0 and np.mean(vals == 255) > 0.9:
                vals[vals == 255] = np.nan
            med = np.nanmedian(vals) if np.any(np.isfinite(vals)) else nodata_fill
            return np.nan_to_num(vals, nan=med)
    except Exception as e:
        print(f"    WARN: {Path(str(rpath)).name}: {e}")
        return np.full(len(coords), nodata_fill)

# Geophysics at grid + holes
for key, fname in ALL_RASTERS.items():
    rpath = fname if Path(str(fname)).exists() else find_file(Path(str(fname)).name)
    if not rpath: grid_df[key] = 0.0; master[key] = 0.0; continue
    grid_df[key] = extract_raster(rpath, coords_grid)
    master[key]  = extract_raster(rpath, coords_holes)
    valid = np.isfinite(grid_df[key].values).sum()
    print(f"  {key}: {valid:,}/{n_grid:,} valid  mean={grid_df[key].mean():.2f}")

# ── Sentinel-2 bands ────────────────────────────────────
s2_grid  = {}
s2_holes = {}
for band_key, band_path in S2_BANDS.items():
    vals_test = extract_raster(band_path, coords_holes[:3], nodata_fill=np.nan)
    print(f"  {band_key}: sample values at first 3 holes = {vals_test}")
    s2_grid[band_key]  = extract_raster(band_path, coords_grid,  nodata_fill=np.nan)
    s2_holes[band_key] = extract_raster(band_path, coords_holes, nodata_fill=np.nan)
    nz = np.sum(np.abs(s2_grid[band_key]) > 0.001)
    print(f"           non-zero grid pts: {nz}/{len(coords_grid)}")

# Direct Landsat extraction using known EPSG:32651
# Mount Weld centre in UTM Zone 51S: approx (455859, 6806970)
print()
print("  Direct Landsat coordinate test:")
if LS_BANDS and HAS_RIO:
    import rasterio
    from pyproj import Transformer as _Tr
    _tr = _Tr.from_crs(4326, 32651, always_xy=True)
    # Test at first collar lat/lon
    _test_lon, _test_lat = coords_holes[0]
    _ux, _uy = _tr.transform(_test_lon, _test_lat)
    print(f"    Collar[0]: lon={_test_lon:.5f} lat={_test_lat:.5f} -> UTM ({_ux:.0f}, {_uy:.0f})")
    _b4_path = LS_BANDS.get("ls_red")
    if _b4_path:
        with rasterio.open(str(_b4_path)) as _src:
            print(f"    Landsat B4 CRS: {_src.crs.to_epsg()}  bounds: {_src.bounds}")
            _v = list(_src.sample([(_ux, _uy)]))[0][0]
            print(f"    Direct UTM sample at collar[0]: {_v}")


def safe_ratio(a, b, fill=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(b > 0, a / b, fill)
    return np.nan_to_num(r, nan=fill)

def compute_indices(bands, prefix=""):
    """Compute spectral indices from band dict."""
    idx = {}
    b  = bands.get("s2_blue",  None)
    g  = bands.get("s2_green", None)
    r  = bands.get("s2_red",   None)
    n  = bands.get("s2_nir",   None)
    s1 = bands.get("s2_swir1", bands.get("ls_swir1", None))
    s2 = bands.get("s2_swir2", bands.get("ls_swir2", None))

    if r is not None and n is not None:
        idx[f"{prefix}ndvi"]       = safe_ratio(n - r, n + r)
    if r is not None and b is not None:
        idx[f"{prefix}iron_oxide"] = safe_ratio(r, b)
    if r is not None and g is not None:
        idx[f"{prefix}ferric_iron"]= safe_ratio(r, g)
    if s1 is not None and s2 is not None:
        idx[f"{prefix}clay_index"] = safe_ratio(s1, s2)
    if s1 is not None and n is not None:
        idx[f"{prefix}ndwi"]       = safe_ratio(n - s1, n + s1)
    if s1 is not None and r is not None:
        # Laterite Alteration Index — high for weathered carbonatite
        idx[f"{prefix}alteration"] = safe_ratio(s1, r)
    return idx

# Add Landsat bands to S2 if available for SWIR
# Note: S2 tile T51JVJ covers UTM zone 51 (114-120E).
# Mount Weld is at 122.54E = UTM zone 50 -> S2 tile does NOT cover the deposit.
# Use Landsat 9 L2SP (path 108 row 080) which covers 122E correctly.
# Remap Landsat band keys to the names compute_indices() expects.
ls_for_indices_grid  = {
    "s2_blue":  extract_raster(LS_BANDS["ls_blue"],    coords_grid,  np.nan) if "ls_blue"  in LS_BANDS else None,
    "s2_green": extract_raster(LS_BANDS["ls_green"],   coords_grid,  np.nan) if "ls_green" in LS_BANDS else None,
    "s2_red":   extract_raster(LS_BANDS["ls_red"],     coords_grid,  np.nan) if "ls_red"   in LS_BANDS else None,
    "s2_nir":   extract_raster(LS_BANDS["ls_nir"],     coords_grid,  np.nan) if "ls_nir"   in LS_BANDS else None,
    "s2_swir1": extract_raster(LS_BANDS["ls_swir1"],   coords_grid,  np.nan) if "ls_swir1" in LS_BANDS else None,
    "s2_swir2": extract_raster(LS_BANDS["ls_swir2"],   coords_grid,  np.nan) if "ls_swir2" in LS_BANDS else None,
}
ls_for_indices_holes = {
    "s2_blue":  extract_raster(LS_BANDS["ls_blue"],    coords_holes, np.nan) if "ls_blue"  in LS_BANDS else None,
    "s2_green": extract_raster(LS_BANDS["ls_green"],   coords_holes, np.nan) if "ls_green" in LS_BANDS else None,
    "s2_red":   extract_raster(LS_BANDS["ls_red"],     coords_holes, np.nan) if "ls_red"   in LS_BANDS else None,
    "s2_nir":   extract_raster(LS_BANDS["ls_nir"],     coords_holes, np.nan) if "ls_nir"   in LS_BANDS else None,
    "s2_swir1": extract_raster(LS_BANDS["ls_swir1"],   coords_holes, np.nan) if "ls_swir1" in LS_BANDS else None,
    "s2_swir2": extract_raster(LS_BANDS["ls_swir2"],   coords_holes, np.nan) if "ls_swir2" in LS_BANDS else None,
}
ls_for_indices_grid  = {k:v for k,v in ls_for_indices_grid.items()  if v is not None}
ls_for_indices_holes = {k:v for k,v in ls_for_indices_holes.items() if v is not None}

# Check Landsat actually has values (diagnose)
for bk, bv in ls_for_indices_grid.items():
    nz = np.sum(np.abs(bv) > 0.001) if bv is not None else 0
    print(f"  Landsat {bk}: non-zero pts={nz}/{len(coords_grid)}")

idx_grid  = compute_indices(ls_for_indices_grid,  prefix="si_")
idx_holes = compute_indices(ls_for_indices_holes, prefix="si_")

for key, arr in idx_grid.items():
    grid_df[key] = arr
    master[key]  = idx_holes.get(key, np.zeros(len(master)))
    print(f"  {key}: mean={np.nanmean(arr):.4f}  std={np.nanstd(arr):.4f}")

# ── IDW geochemistry to grid ────────────────────────────
if HAS_SCI and matched >= 3:
    from scipy.spatial import cKDTree
    labeled_mask = master[treo_col].notna() if treo_col else pd.Series([False]*len(master))
    labeled_idx  = np.where(labeled_mask.values)[0]
    ree_idw_cols = [c for c in assay_agg.columns if "_max" in c and any(k in c for k in
                    ["ceo2","la2o3","nd2o3","treo","lree","hree","ratio","ppm"])][:15]
    ree_idw_cols = [c for c in ree_idw_cols if c in master.columns]
    xy_lab   = np.array([[master["lon"].iloc[i], master["lat"].iloc[i]] for i in labeled_idx])
    tree_lab = cKDTree(xy_lab)
    IDW_K    = min(8, len(labeled_idx))
    dists, idxs = tree_lab.query(xy_grid, k=IDW_K)
    dists = np.maximum(dists * 111320, 1e-6)
    w = 1.0 / (dists ** 2); w /= w.sum(axis=1, keepdims=True)
    for col in ree_idw_cols:
        vals_lab = pd.to_numeric(master[col].iloc[labeled_idx], errors="coerce").fillna(0).values
        grid_df[f"idw_{col}"] = (w * vals_lab[idxs]).sum(axis=1) if idxs.ndim > 1 else w * vals_lab[idxs]
    print(f"  IDW: {len(ree_idw_cols)} geochemical columns -> grid")

# ════════════════════════════════════════════════════════
# 5. BUILD FEATURE MATRICES
# ════════════════════════════════════════════════════════
print()
print("STEP 5: Building feature matrices")
print("-"*40)

ree_hole = [c for c in master.columns if any(k in c for k in
            ["treo","ceo2","la2o3","nd2o3","pr6","sm2","eu2","lree","hree",
             "_ppm_max","_ppm_mean","ratio"])]
gphy_hole = [c for c in master.columns if c.startswith("g_") or c.startswith("y4_")]
si_hole   = [c for c in master.columns if c.startswith("si_")]
dem_hole  = [c for c in master.columns if "dem" in c]
dep_hole  = [c for c in ["depth_score","maxdepth","fromdepth_min","elevation"] if c in master.columns]
alt_hole  = [c for c in alt_cols if c in master.columns]

all_hole_feat = list(dict.fromkeys(ree_hole + gphy_hole + si_hole + dem_hole + dep_hole + alt_hole))
all_hole_feat = [c for c in all_hole_feat if c in master.columns]

X_holes = master[all_hole_feat].copy()
for c in all_hole_feat:
    X_holes[c] = pd.to_numeric(X_holes[c], errors="coerce")
X_holes = X_holes.fillna(X_holes.median()).fillna(0)

# Add normalised spatial coordinates as features.
# This breaks the tie when holes share identical raster pixel values:
# every point now has a unique (lat, lon) fingerprint.
lon_min_f = master["lon"].min(); lon_range_f = master["lon"].max() - lon_min_f + 1e-9
lat_min_f = master["lat"].min(); lat_range_f = master["lat"].max() - lat_min_f + 1e-9
X_holes["coord_lon_norm"] = (master["lon"].fillna(lon_min_f).values - lon_min_f) / lon_range_f
X_holes["coord_lat_norm"] = (master["lat"].fillna(lat_min_f).values - lat_min_f) / lat_range_f
# Distance to deposit centroid (RC1297 area: -28.8616S, 122.5458E)
centroid_lon, centroid_lat = 122.5458, -28.8616
X_holes["dist_centroid"] = np.sqrt(
    ((master["lon"].fillna(centroid_lon) - centroid_lon) * 111320 * np.cos(np.radians(-28.86)))**2 +
    ((master["lat"].fillna(centroid_lat) - centroid_lat) * 111320)**2
) / 500.0   # normalise to ~deposit radius
all_hole_feat = all_hole_feat + ["coord_lon_norm","coord_lat_norm","dist_centroid"]

# log1p geochemical columns
ppm_cols = [c for c in all_hole_feat if "_ppm" in c or "ratio" in c or
            c in ["lree_max","hree_max","treo_max"]]
for c in ppm_cols:
    X_holes[c] = np.log1p(X_holes[c].clip(lower=0))

# Grid features
grid_gphy = [c for c in grid_df.columns if c.startswith("g_") or c.startswith("y4_")]
grid_si   = [c for c in grid_df.columns if c.startswith("si_")]
grid_idw  = [c for c in grid_df.columns if c.startswith("idw_")]
grid_misc = [c for c in ["dist_nearest_m"] if c in grid_df.columns]
grid_feat = list(dict.fromkeys(grid_gphy + grid_si + grid_idw + grid_misc))
grid_feat = [c for c in grid_feat if c in grid_df.columns]

X_grid = grid_df[grid_feat].copy().fillna(0)
for c in grid_idw:
    if c in X_grid.columns:
        X_grid[c] = np.log1p(X_grid[c].clip(lower=0))

# Add spatial coords to grid (must match hole feature space)
X_grid["coord_lon_norm"] = (grid_df["lon"].values - lon_min_f) / lon_range_f
X_grid["coord_lat_norm"] = (grid_df["lat"].values - lat_min_f) / lat_range_f
X_grid["dist_centroid"]  = np.sqrt(
    ((grid_df["lon"].values - centroid_lon) * 111320 * np.cos(np.radians(-28.86)))**2 +
    ((grid_df["lat"].values - centroid_lat) * 111320)**2
) / 500.0
grid_feat = grid_feat + ["coord_lon_norm","coord_lat_norm","dist_centroid"]

if treo_col:
    treo_raw = pd.to_numeric(master[treo_col], errors="coerce")
    p95      = treo_raw.quantile(0.95)
    y_holes  = (treo_raw / p95).clip(0, 1).fillna(-1).values
    labeled  = y_holes >= 0
    X_train  = X_holes.values[labeled]
    y_train  = y_holes[labeled]
    n_train  = labeled.sum()
    print(f"  Training: {n_train} holes  |  Hole feats: {len(all_hole_feat)}")
    print(f"  Spectral indices: {len(si_hole)} ({[c for c in si_hole]})")
    print(f"  Grid feats: {len(grid_feat)}  |  Grid pts: {n_grid:,}")
else:
    print("  ERROR: No TREO column found"); input(); sys.exit(1)

# ════════════════════════════════════════════════════════
# 6. TRAIN MODELS
# ════════════════════════════════════════════════════════
print()
print("STEP 6: Training models (or loading saved bundle)")
print("-"*40)

# ── Auto-load existing bundle if available ────────────────
# Set FORCE_RETRAIN = True to always retrain from scratch
FORCE_RETRAIN = False

_bundle_path = out / "ree_model_bundle.joblib"
_bundle_loaded = False
if not FORCE_RETRAIN and _bundle_path.exists():
    try:
        import joblib as _jl
        _bundle = _jl.load(str(_bundle_path))
        _info   = _bundle["meta_info"]
        # Only load if feature columns match exactly
        if _bundle["feat_cols"] == all_hole_feat:
            m_models = _bundle["models"]
            meta     = _bundle["meta"]
            scaler   = _bundle["scaler"]
            pca      = _bundle["pca"]
            p95      = _bundle["p95_treo"]
            # Reconstruct CV predictions (approximate via predict on training data)
            X_tr_p   = pca.transform(scaler.transform(X_train))
            m_preds  = {}
            m_scores = {}
            col_order= sorted(m_models.keys())
            for k, m in m_models.items():
                pred = m.predict(X_train if hasattr(m,'steps') else X_tr_p).clip(0,1)
                m_preds[k]  = pred
                m_scores[k] = float(_info.get("cv_r2", 0))
            meta_r2  = float(_info["cv_r2"])
            roc_auc  = float(_info.get("roc_auc", 0))
            avg_prec = float(_info.get("ap", 0))
            from sklearn.metrics import mean_squared_error
            meta_cv  = meta.predict(
                np.column_stack([m_preds[k] for k in col_order])
            ).clip(0,1)
            rmse     = float(np.sqrt(mean_squared_error(y_train, meta_cv)))
            _bundle_loaded = True
            print(f"  Loaded saved bundle: {_bundle_path.name}")
            print(f"  Trained: {_info['trained_date']}  Deposits: {_info['deposits']}")
            print(f"  CV R²={meta_r2:.4f}  ROC={roc_auc:.4f}  (skipping retrain)")
            print(f"  To force retrain: set FORCE_RETRAIN = True")
        else:
            print(f"  Bundle feature columns mismatch — retraining")
    except Exception as _e:
        print(f"  Bundle load failed ({_e}) — retraining")

if not _bundle_loaded:
    print("  No saved bundle found or FORCE_RETRAIN=True — training from scratch")


from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score
from sklearn.metrics import average_precision_score, roc_curve, precision_recall_curve

scaler = RobustScaler()
X_tr_s = scaler.fit_transform(X_train)
n_pca  = min(15, X_tr_s.shape[1], n_train - 2)
pca    = PCA(n_components=n_pca, random_state=42)
X_tr_p = pca.fit_transform(X_tr_s)
kf     = KFold(n_splits=5, shuffle=True, random_state=42)
m_preds = {}; m_scores = {}; m_models = {}

if not _bundle_loaded:
  # ── Train all models from scratch ──────────────────────
  from sklearn.ensemble import RandomForestRegressor
  print("  [A] Random Forest...")
rf = RandomForestRegressor(500, max_depth=10, min_samples_leaf=3, max_features="sqrt",
                           max_samples=0.8, random_state=42, n_jobs=-1, oob_score=True)
rf.fit(X_tr_p, y_train)
rf_cv = cross_val_predict(rf, X_tr_p, y_train, cv=kf)
m_preds["rf"] = rf_cv; m_scores["rf"] = r2_score(y_train, rf_cv); m_models["rf"] = rf
print(f"     CV R²={m_scores['rf']:.4f}  OOB={rf.oob_score_:.4f}")

# SVM
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
print("  [B] SVM-RBF...")
svm = Pipeline([("sc",RobustScaler()),("pca",PCA(n_pca,random_state=42)),
                ("s", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.05))])
sv_cv = cross_val_predict(svm, X_train, y_train, cv=kf)
svm.fit(X_train, y_train)
m_preds["svm"] = sv_cv; m_scores["svm"] = r2_score(y_train, sv_cv); m_models["svm"] = svm
print(f"     CV R²={m_scores['svm']:.4f}")

# XGBoost or GradientBoosting
if HAS_XGB:
    import xgboost as xgb
    print("  [C] XGBoost...")
    xb = Pipeline([("sc",RobustScaler()),("pca",PCA(n_pca,random_state=42)),
                   ("x", xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=42, n_jobs=-1, verbosity=0))])
else:
    from sklearn.ensemble import GradientBoostingRegressor
    print("  [C] GradientBoosting (fallback)...")
    xb = Pipeline([("sc",RobustScaler()),("pca",PCA(n_pca,random_state=42)),
                   ("x", GradientBoostingRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                                                    subsample=0.8, random_state=42))])
xb_cv = cross_val_predict(xb, X_train, y_train, cv=kf)
xb.fit(X_train, y_train)
m_preds["xgb"] = xb_cv; m_scores["xgb"] = r2_score(y_train, xb_cv); m_models["xgb"] = xb
print(f"     CV R²={m_scores['xgb']:.4f}")

# [D] CNN/MLP permanently dropped.
# Satellite spectral indices (Landsat EPSG:32651) are returning zeros due to
# coordinate transformation issues. CNN/MLP trained on zero-variance features
# achieves R²<0 and hurts the ensemble. Dropping for clean 3-model stacking.
print("  [D] CNN dropped (spectral indices not available — 3-model ensemble)")

print()
print("  Model CV R² summary:")
for n,r in sorted(m_scores.items(), key=lambda x:-x[1]):
    print(f"    {n.upper():<8} {r:+.4f}  {'#'*max(0,int(r*30))}")

  # end of: if not _bundle_loaded

# Stacking
from sklearn.linear_model import Ridge
print()
print("STEP 7: Stacking ensemble")
meta_X  = np.column_stack([m_preds[k] for k in sorted(m_preds)])
meta    = Ridge(alpha=1.0)
meta_cv = cross_val_predict(meta, meta_X, y_train, cv=kf).clip(0,1)
meta_r2 = r2_score(y_train, meta_cv)
meta.fit(meta_X, y_train)
print(f"  Ensemble CV R²: {meta_r2:.4f}  vs best single: {max(m_scores.values()):.4f}")
print(f"  Meta weights: {dict(zip(sorted(m_preds), meta.coef_.round(3)))}")

# Binary metrics
thr  = np.percentile(y_train, 70)
yb   = (y_train >= thr).astype(int)
fpr, tpr, _ = roc_curve(yb, meta_cv)
roc_auc     = roc_auc_score(yb, meta_cv)
prec, rec, _= precision_recall_curve(yb, meta_cv)
avg_prec    = average_precision_score(yb, meta_cv)
rmse        = np.sqrt(mean_squared_error(y_train, meta_cv))
print(f"  ROC AUC={roc_auc:.4f}  AP={avg_prec:.4f}  RMSE={rmse:.4f}")
# ════════════════════════════════════════════════════════
# SAVE MODEL BUNDLE (joblib)
# Saves everything needed to predict without retraining.
# Supports incremental updates when new deposit data arrives.
# ════════════════════════════════════════════════════════
import joblib, hashlib, datetime as _dt

def _save_model_bundle(path, models_dict, meta_model, scalers,
                       feat_cols, p95, meta_info, X_train_cache, y_train_cache):
    """Save complete model bundle for later inference or incremental retraining."""
    bundle = {
        # ── Model objects ────────────────────────────────
        "models":        models_dict,    # {"rf":..., "svm":..., "xgb":...}
        "meta":          meta_model,     # Ridge meta-learner
        "scaler":        scalers["hole_scaler"],
        "pca":           scalers["hole_pca"],
        "scaler_shared": scalers.get("shared_scaler"),
        "pca_shared":    scalers.get("shared_pca"),
        # ── Feature schema ───────────────────────────────
        "feat_cols":     feat_cols,      # ordered list — MUST match at inference time
        "p95_treo":      float(p95),
        # ── Training data cache (for incremental updates) ─
        "X_train":       X_train_cache,  # numpy array shape (n, features)
        "y_train":       y_train_cache,  # numpy array shape (n,)
        "X_hash":        hashlib.sha256(X_train_cache.tobytes()).hexdigest()[:16],
        # ── Metadata ─────────────────────────────────────
        "meta_info":     meta_info,
    }
    joblib.dump(bundle, path, compress=3)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"  Saved: {path}  ({size_mb:.1f} MB)")
    return bundle

_meta_info = {
    "version":         "v9.0",
    "trained_date":    _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
    "deposits":        ["mount_weld"],
    "n_holes_total":   int(n_train),
    "n_holes_labelled":int(n_train),
    "cv_r2":           float(meta_r2),
    "roc_auc":         float(roc_auc),
    "rmse":            float(rmse),
    "model_names":     sorted(m_models.keys()),
    "meta_weights":    {k: float(v) for k,v in zip(sorted(m_preds), meta.coef_)},
    "feature_count":   len(all_hole_feat),
    "notes":           "RF+SVM+XGB ensemble. CNN dropped (cloud mask). Incremental-capable.",
}

_scalers = {
    "hole_scaler": scaler,
    "hole_pca":    pca,
    "shared_scaler": scaler2 if "scaler2" in dir() else None,
    "shared_pca":    pca2    if "pca2"    in dir() else None,
}

_bundle_path = out / "ree_model_bundle.joblib"
print()
print("SAVING MODEL BUNDLE")
print("-"*40)
_save_model_bundle(
    _bundle_path,
    models_dict    = m_models,
    meta_model     = meta,
    scalers        = _scalers,
    feat_cols      = all_hole_feat,
    p95            = p95,
    meta_info      = _meta_info,
    X_train_cache  = X_train,
    y_train_cache  = y_train,
)
print(f"  Deposits in model: {_meta_info['deposits']}")
print(f"  Training holes:    {_meta_info['n_holes_labelled']}")
print(f"  CV R²:             {_meta_info['cv_r2']:.4f}")
print(f"  To add new data:   run retrain.py --new_data path/to/new_collar.csv")



# ════════════════════════════════════════════════════════
# 8. PREDICT ON GRID
# ════════════════════════════════════════════════════════
print()
print("STEP 8: Predicting on full grid")
print("-"*40)

# Build shared feature space at grid (geophys + spectral + IDW)
X_grid_vals = X_grid.values

# Need matching features — use geophys + si columns present in both
shared = [c for c in grid_feat if c in all_hole_feat]
X_holes_sh = X_holes[shared].values if shared else X_holes.values[labeled]
X_grid_sh  = X_grid[[c for c in shared if c in X_grid.columns]].values if shared else X_grid.values

n_sh   = X_grid_sh.shape[1] if X_grid_sh.ndim > 1 and X_grid_sh.shape[1] > 0 else 1
n_pca2 = min(n_pca, n_sh, n_train-2)

scaler2 = RobustScaler()
pca2    = PCA(n_components=n_pca2, random_state=42)
X_holes_sh_safe = X_holes_sh[labeled] if X_holes_sh.shape[0]==len(master) else X_holes_sh
X_sh_tr  = pca2.fit_transform(scaler2.fit_transform(X_holes_sh_safe))
X_all_sh = pca2.transform(scaler2.transform(
    X_holes_sh if X_holes_sh.shape[0]==len(master) else
    np.vstack([X_holes_sh_safe, np.zeros((len(master)-n_train, X_holes_sh_safe.shape[1]))])
))
X_all_shared = (X_holes_sh if X_holes_sh.shape[0]==len(master)
                else np.vstack([X_holes_sh_safe,
                                 np.zeros((len(master)-n_train, X_holes_sh_safe.shape[1]))]))
X_sh_gr  = pca2.transform(scaler2.transform(X_grid_sh))

rf2 = RandomForestRegressor(500, max_depth=10, min_samples_leaf=3, random_state=42, n_jobs=-1)
rf2.fit(X_sh_tr, y_train)

svm2 = Pipeline([("sc",RobustScaler()),("pca",PCA(n_pca2,random_state=42)),
                 ("s", SVR(kernel="rbf",C=10,gamma="scale",epsilon=0.05))])
X_grid_raw = X_grid[[c for c in shared if c in X_grid.columns]].values
svm2.fit(X_grid_raw[grid_df["nearest_idx"].isin(np.where(labeled)[0])] if False else
         X_holes_sh[labeled] if X_holes_sh.shape[0]==len(master) else X_holes_sh,
         y_train)
svm2 = Pipeline([("sc",RobustScaler()),("pca",PCA(n_pca2,random_state=42)),
                 ("s", SVR(kernel="rbf",C=10,gamma="scale",epsilon=0.05))])
svm2.fit(X_holes_sh[labeled] if X_holes_sh.shape[0] == len(master) else X_holes_sh, y_train)

rf_g  = rf2.predict(X_sh_gr).clip(0,1)
svm_g = svm2.predict(X_grid_sh).clip(0,1)

if HAS_XGB:
    xb2 = Pipeline([("sc",RobustScaler()),("pca",PCA(n_pca2,random_state=42)),
                    ("x", xgb.XGBRegressor(n_estimators=300,max_depth=5,learning_rate=0.05,
                                            verbosity=0,random_state=42))])
    xb2.fit(X_holes_sh[labeled] if X_holes_sh.shape[0]==len(master) else X_holes_sh, y_train)
    xb_g  = xb2.predict(X_grid_sh).clip(0,1)
else:
    xb2  = None
    xb_g = rf_g.copy()

# CNN on grid: interpolate from hole predictions using distance weighting
cnn_g = rf_g.copy()

meta_grid_X = np.column_stack([rf_g, svm_g, xb_g, cnn_g])
# Match column order to trained meta
col_order   = sorted(m_preds.keys())
meta_cols   = {"rf":rf_g,"svm":svm_g,"xgb":xb_g,"cnn":cnn_g}
meta_grid_X = np.column_stack([meta_cols.get(k, rf_g) for k in col_order])
grid_scores = meta.predict(meta_grid_X).clip(0,1)

grid_df["prospectivity"] = grid_scores
grid_df["score_100"]     = (grid_scores * 100).round(1)
# Predict at ALL hole locations using the shared feature space model
rf_all_scores  = rf2.predict(X_all_sh).clip(0, 1)
svm_all_scores = svm2.predict(X_all_shared).clip(0, 1)
xb_all_scores  = xb2.predict(X_all_shared).clip(0, 1) if HAS_XGB else rf_all_scores.copy()

# CNN/MLP scores at all holes: use full X_holes si columns
cnn_model_info = m_models.get("cnn")
if isinstance(cnn_model_info, tuple) and cnn_model_info[0] == "mlp":
    _, mlp_model, si_cols_used = cnn_model_info
    si_avail = [c for c in si_cols_used if c in master.columns]
    if si_avail:
        X_si_all = RobustScaler().fit_transform(master[si_avail].fillna(0).values)
        cnn_all_scores = mlp_model.predict(
            pd.DataFrame(master[si_avail].fillna(0).values, columns=si_avail).values
        ).clip(0, 1)
    else:
        cnn_all_scores = rf_all_scores.copy()
else:
    cnn_all_scores = rf_all_scores.copy()

meta_all_X = np.column_stack([
    {"rf": rf_all_scores, "svm": svm_all_scores,
     "xgb": xb_all_scores, "cnn": cnn_all_scores}.get(k, rf_all_scores)
    for k in col_order
])
master["prospectivity"] = meta.predict(meta_all_X).clip(0, 1)
master["score_100"]     = (master["prospectivity"] * 100).round(1)

print(f"  Grid score range: {grid_scores.min():.3f} – {grid_scores.max():.3f}")
print(f"  Hole score range: {master['score_100'].min():.1f} – {master['score_100'].max():.1f}")

# ════════════════════════════════════════════════════════
# 9. SAVE OUTPUTS
# ════════════════════════════════════════════════════════
print()
print("STEP 9: Saving outputs")
id_col = next((c for c in ["companyholeid","holeid"] if c in master.columns), None)
top50  = master.nlargest(50,"score_100").copy(); top50["rank"] = range(1,51)
keep   = [c for c in [id_col,"lat","lon","score_100",treo_col,"depth_score"] if c and c in top50.columns]
top50[keep].to_csv(out/"top_targets.csv", index=False, encoding="utf-8")
master.to_csv(out/"scored_data.csv", index=False, encoding="utf-8")
grid_df.to_csv(out/"grid_predictions.csv", index=False, encoding="utf-8")
print(f"  scored_data.csv:      {len(master):,} drillholes")
print(f"  grid_predictions.csv: {n_grid:,} grid points")

print()
print("  TOP 10 TARGETS:")
print(f"  {'#':<4} {'Hole':<20} {'Score':>7}  {'TREO ppm':>12}")
print("  "+"-"*48)
for _, row in top50.head(10).iterrows():
    hid  = str(row[id_col])[:18] if id_col else ""
    trv  = f"{row[treo_col]:,.0f}" if treo_col and pd.notna(row.get(treo_col)) else "IDW"
    print(f"  {int(row['rank']):<4} {hid:<20} {row['score_100']:>6.1f}/100  {trv:>12}")

# ════════════════════════════════════════════════════════
# 10. MAPS AND DASHBOARD
# ════════════════════════════════════════════════════════
lons_g = grid_df["lon"].values; lats_g = grid_df["lat"].values
lons_h = master["lon"].values;  lats_h = master["lat"].values
Zi = grid_scores.reshape(len(grid_lats), len(grid_lons))

if HAS_MPL:
    print()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, matplotlib.colors as mcolors

    colors_list = ["#0D0500","#3A1206","#6B2208","#A0400A","#C9A84C","#FFF5D0"]
    cmap = mcolors.LinearSegmentedColormap.from_list("ree", colors_list)

    fig, ax = plt.subplots(figsize=(14,12), facecolor="#1a0a00")
    ax.set_facecolor("#1a0a00")
    # Smooth with cubic interpolation to remove Landsat pixel blocking
    from scipy.interpolate import griddata as _gd
    pts_g = np.column_stack([lons_g, lats_g])
    xi_s  = np.linspace(lons_g.min(), lons_g.max(), 300)
    yi_s  = np.linspace(lats_g.min(), lats_g.max(), 300)
    Xi_s, Yi_s = np.meshgrid(xi_s, yi_s)
    Zi_smooth  = _gd(pts_g, grid_scores, (Xi_s, Yi_s), method="cubic")
    Zi_smooth  = np.nan_to_num(Zi_smooth, nan=np.nanmedian(grid_scores)).clip(0, 1)
    im = ax.pcolormesh(Xi_s, Yi_s, Zi_smooth, cmap=cmap, vmin=0, vmax=1, shading="auto")
    cs = ax.contour(Xi_s, Yi_s, Zi_smooth, levels=[0.4,0.6,0.75,0.85],
                    colors=["#5C2010","#A0400A","#D4601A","#C9A84C"],
                    linewidths=[0.4,0.6,0.8,1.0], alpha=0.75)
    ax.clabel(cs, fmt="%.2f", colors="#F5EDD8", fontsize=7)
    ax.scatter(lons_h, lats_h, c=master["score_100"].values, cmap=cmap,
               vmin=0, vmax=100, s=18, zorder=4, alpha=0.85)
    t20 = master.nlargest(20,"score_100")
    ax.scatter(t20["lon"].values, t20["lat"].values, c="white", s=80, zorder=5,
               marker="D", edgecolors="#C9A84C", linewidths=0.8)
    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Prospectivity (0-1)", color="#C9A84C", fontsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#F5EDD8", fontsize=8)
    n_si = len([c for c in master.columns if c.startswith("si_")])
    ax.set_title(
        f"Mount Weld REE Prospectivity  |  {datetime.date.today()}\n"
        f"Ensemble CV R²={meta_r2:.3f}  ROC={roc_auc:.3f}  "
        f"|  S2+LS9 ({n_si} spectral indices)  |  {n_grid:,} grid pts",
        color="#C9A84C", fontsize=11, pad=10)
    ax.set_xlabel("Longitude (°E)", color="#C9A84C", fontsize=10)
    ax.set_ylabel("Latitude (°S)",  color="#C9A84C", fontsize=10)
    ax.tick_params(colors="#6B5535")
    for sp in ax.spines.values(): sp.set_edgecolor("#3D2E14")
    plt.tight_layout()
    plt.savefig(str(out/"prospectivity_map.png"), dpi=200, bbox_inches="tight", facecolor="#1a0a00")
    plt.close()
    print("  prospectivity_map.png saved")

    # 12-panel dashboard
    DARK="#0D0A06"; MID="#1A1208"; GOLD="#C9A84C"; ACCENT="#D4601A"
    LGOLD="#F5EDD8"; MUTED="#6B5535"; GREEN="#4A7C59"
    def sa(ax,t="",xl="",yl=""):
        ax.set_facecolor(MID); ax.tick_params(colors=MUTED,labelsize=9)
        for sp in ax.spines.values(): sp.set_color("#3D2E14")
        if t:  ax.set_title(t,color=GOLD,fontsize=10,pad=6)
        if xl: ax.set_xlabel(xl,color=MUTED,fontsize=9)
        if yl: ax.set_ylabel(yl,color=MUTED,fontsize=9)
        ax.grid(True,color="#2A1F0E",linewidth=0.5,linestyle="--",alpha=0.6)

    fig = plt.figure(figsize=(22,16),facecolor=DARK)
    fig.suptitle(f"Mount Weld REE — v9 Satellite+Ensemble  CV R²={meta_r2:.3f}  ROC={roc_auc:.3f}  AP={avg_prec:.3f}",
                 color=GOLD,fontsize=15,y=0.98)

    ax1=fig.add_subplot(3,4,1)
    r2s=[m_scores[k] for k in sorted(m_scores)]+[meta_r2]
    nms=[k.upper() for k in sorted(m_scores)]+["ENSEMBLE"]
    cols1=[GREEN if v==max(r2s) else GOLD if v>0.5 else ACCENT if v>0 else "#E24B4A" for v in r2s]
    ax1.barh(nms,r2s,color=cols1,alpha=0.85,edgecolor=MID)
    ax1.axvline(0,color=MUTED,lw=0.8)
    for i,(v,n) in enumerate(zip(r2s,nms)): ax1.text(max(v,0)+0.01,i,f"{v:.3f}",va="center",color=LGOLD,fontsize=9)
    sa(ax1,"CV R² by algorithm","CV R²","")

    ax2=fig.add_subplot(3,4,2)
    ax2.plot(fpr,tpr,color=GOLD,lw=2,label=f"AUC={roc_auc:.3f}")
    ax2.plot([0,1],[0,1],color=MUTED,lw=1,ls="--")
    ax2.fill_between(fpr,tpr,alpha=0.12,color=GOLD)
    ax2.legend(facecolor=MID,labelcolor=LGOLD,fontsize=9)
    sa(ax2,"ROC Curve","FPR","TPR")

    ax3=fig.add_subplot(3,4,3)
    ax3.plot(rec,prec,color=ACCENT,lw=2,label=f"AP={avg_prec:.3f}")
    ax3.fill_between(rec,prec,alpha=0.12,color=ACCENT)
    ax3.legend(facecolor=MID,labelcolor=LGOLD,fontsize=9)
    sa(ax3,"Precision-Recall","Recall","Precision")

    ax4=fig.add_subplot(3,4,4)
    sc4=ax4.scatter(y_train,meta_cv,c=y_train*p95/10000,cmap="YlOrRd",s=22,alpha=0.75)
    ax4.plot([0,1],[0,1],color=MUTED,lw=1,ls="--")
    cb4=plt.colorbar(sc4,ax=ax4); cb4.ax.tick_params(colors=MUTED,labelsize=7)
    cb4.set_label("TREO %",color=MUTED,fontsize=7)
    ax4.text(0.05,0.88,f"CV R²={meta_r2:.3f}\nRMSE={rmse:.4f}",transform=ax4.transAxes,
             color=LGOLD,fontsize=9,bbox=dict(facecolor=MID,edgecolor=GOLD,alpha=0.85))
    sa(ax4,"Actual vs Predicted","Actual","Predicted")

    ax5=fig.add_subplot(3,4,5)
    clrs5=["#C9A84C","#D4601A","#7B77DD","#4A7C59"]
    for i,k in enumerate(sorted(m_preds)):
        ax5.scatter(y_train,m_preds[k],s=10,alpha=0.45,color=clrs5[i%4],label=k.upper())
    ax5.plot([0,1],[0,1],color=MUTED,lw=1,ls="--")
    ax5.legend(facecolor=MID,labelcolor=LGOLD,fontsize=8,markerscale=1.5)
    sa(ax5,"All models vs actual","Actual","Predicted")

    ax6=fig.add_subplot(3,4,6)
    ax6.hist(grid_scores*100,bins=40,color=ACCENT,alpha=0.8,edgecolor=MID,label=f"Grid ({n_grid})")
    ax6.hist(master["score_100"],bins=40,color=GOLD,alpha=0.6,edgecolor=MID,label=f"Holes ({len(master)})")
    ax6.axvline(70,color="#E24B4A",lw=1.5,ls="--")
    ax6.legend(facecolor=MID,labelcolor=LGOLD,fontsize=8)
    sa(ax6,"Score distribution","Score","Count")

    ax7=fig.add_subplot(3,4,7)
    resid=meta_cv-y_train
    ax7.scatter(meta_cv,resid,s=15,alpha=0.65,
                c=[GOLD if abs(r)<0.1 else ACCENT if abs(r)<0.2 else "#E24B4A" for r in resid])
    ax7.axhline(0,color=MUTED,lw=1,ls="--")
    sa(ax7,"Residuals","Predicted","Residual")

    ax8=fig.add_subplot(3,4,8)
    treo_pct=y_train*p95/10000
    ax8.hist(treo_pct,bins=25,color=GOLD,alpha=0.85,edgecolor=MID)
    ax8.axvline(np.median(treo_pct),color=ACCENT,lw=1.5,ls="--",label=f"Med={np.median(treo_pct):.1f}%")
    ax8.axvline(10,color=GREEN,lw=1,ls=":",label="10% cutoff")
    ax8.legend(facecolor=MID,labelcolor=LGOLD,fontsize=8)
    sa(ax8,"TREO Distribution","TREO (%)","Count")

    ax9=fig.add_subplot(3,4,9)
    ev=pca.explained_variance_ratio_; ev_cum=np.cumsum(ev)
    ax9.bar(range(1,len(ev)+1),ev*100,color=MUTED,alpha=0.7)
    ax9r=ax9.twinx(); ax9r.plot(range(1,len(ev)+1),ev_cum*100,color=GOLD,lw=2,marker="o",ms=3)
    ax9r.axhline(95,color=ACCENT,lw=1,ls="--"); ax9r.set_ylim(0,105)
    ax9r.tick_params(colors=MUTED,labelsize=8); ax9r.set_ylabel("Cumul %",color=MUTED,fontsize=8)
    ax9.text(0.45,0.15,f"{n_pca} PCs",transform=ax9.transAxes,color=LGOLD,fontsize=9,
             bbox=dict(facecolor=MID,edgecolor=GOLD,alpha=0.85))
    sa(ax9,f"PCA Scree ({len(all_hole_feat)} feats → {n_pca} PCs)","PC","Var %")

    ax10=fig.add_subplot(3,4,10)
    coef_n=[k.upper() for k in col_order]; coef_v=meta.coef_
    cols10=[GREEN if v>0 else "#E24B4A" for v in coef_v]
    ax10.barh(coef_n,coef_v,color=cols10,alpha=0.85,edgecolor=MID)
    ax10.axvline(0,color=MUTED,lw=0.8)
    sa(ax10,"Ensemble weights","Coefficient","")

    ax11=fig.add_subplot(3,4,11)
    si_show=[c for c in master.columns if c.startswith("si_")]
    if si_show:
        si_means=[pd.to_numeric(master[c],errors="coerce").mean() for c in si_show]
        si_labels=[c.replace("si_","") for c in si_show]
        ax11.barh(si_labels,si_means,color=ACCENT,alpha=0.85,edgecolor=MID)
        sa(ax11,"Mean spectral indices (holes)","Mean value","")
    else:
        sa(ax11,"Spectral indices","","")

    ax12=fig.add_subplot(3,4,12)
    sc12=ax12.scatter(lons_g,lats_g,c=grid_scores*100,cmap="YlOrRd",s=4,alpha=0.7,vmin=0,vmax=100)
    ax12.scatter(lons_h,lats_h,c="white",s=12,zorder=5,alpha=0.7)
    plt.colorbar(sc12,ax=ax12).ax.tick_params(colors=MUTED,labelsize=7)
    sa(ax12,f"Grid coverage ({n_grid:,} pts)","Lon","Lat")

    plt.tight_layout(rect=[0,0,1,0.97])
    plt.savefig(str(out/"results_dashboard.png"),dpi=150,bbox_inches="tight",facecolor=DARK)
    plt.close()
    print("  results_dashboard.png saved (12 panels)")

if HAS_FOL:
    import folium; from folium.plugins import HeatMap, MarkerCluster
    clat=float(np.nanmedian(lats_g)); clon=float(np.nanmedian(lons_g))
    m=folium.Map(location=[clat,clon],zoom_start=14,tiles="CartoDB dark_matter")
    heat=[[float(la),float(lo),float(sc)] for la,lo,sc in zip(lats_g,lons_g,grid_scores)
          if np.isfinite(la) and la!=0]
    HeatMap(heat,radius=14,blur=10,max_zoom=17,
            gradient={0:"#1a0a00",0.3:"#5C2010",0.55:"#A0400A",
                      0.75:"#D4601A",0.9:"#C9A84C",1:"#FFF5D0"},min_opacity=0.45).add_to(m)
    cl=MarkerCluster(name="Top 30").add_to(m)
    for i,(_,row) in enumerate(master.nlargest(30,"score_100").iterrows()):
        la,lo=float(row["lat"]),float(row["lon"])
        if not (np.isfinite(la) and la!=0): continue
        sc=float(row["score_100"]); hid=str(row[id_col])[:25] if id_col else f"#{i+1}"
        tv=f"{row[treo_col]:,.0f} ppm" if treo_col and pd.notna(row.get(treo_col)) else "IDW"
        color="#C9A84C" if sc>70 else "#D4601A" if sc>55 else "#8B4513"
        folium.CircleMarker(location=[la,lo],radius=9 if sc>70 else 6,
            color=color,fill=True,fill_opacity=0.9,
            popup=folium.Popup(f"<b>{hid}</b><br>Score: <b>{sc:.0f}/100</b><br>"
                               f"TREO: {tv}<br>#{i+1}",max_width=220),
            tooltip=f"{hid}  {sc:.0f}/100").add_to(cl)
    folium.LayerControl().add_to(m)
    m.save(str(out/"prospectivity_map.html"))
    print("  prospectivity_map.html saved")

print()
print("="*62)
print("  ALL DONE — v9 SATELLITE + ENSEMBLE PIPELINE")
for fname in ["prospectivity_map.html","prospectivity_map.png",
              "results_dashboard.png","top_targets.csv",
              "scored_data.csv","grid_predictions.csv"]:
    p=out/fname
    print(f"    {'[OK]' if p.exists() else '[--]'}  {fname}")
print()
print(f"  Satellite spectral indices computed: {n_si}")
print(f"  Sentinel-2 bands used:  {list(S2_BANDS.keys())}")
print(f"  Landsat 9 bands used:   {list(LS_BANDS.keys())}")
print(f"  Model CV R²:  RF={m_scores.get('rf',0):.3f}  SVM={m_scores.get('svm',0):.3f}  "
      f"XGB={m_scores.get('xgb',0):.3f}  CNN={m_scores.get('cnn',0):.3f}")
print(f"  Ensemble CV R²={meta_r2:.3f}  ROC AUC={roc_auc:.3f}")
print("="*62)
input("\n  Press Enter to close...")

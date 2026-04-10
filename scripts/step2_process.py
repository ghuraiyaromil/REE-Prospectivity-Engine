"""
REE PROSPECTIVITY ENGINE -- STEP 2: PROCESS DATA
Customised for drillhole data processing.
"""
import sys
import warnings
import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root to path for geoai import
sys.path.append(str(Path(__file__).parent.parent))
from geoai.config import DEPOSITS_FOLDER, OUTPUT_DIR as OUTPUT_FOLDER

BASE_EXTRACTED = DEPOSITS_FOLDER / "extracted"
BASE_ORIGINAL  = DEPOSITS_FOLDER

RASTER_TARGETS = {
    'tmi':          'magmap_v7_2019_TMI_ed_VRTP_05VD_geotiff.tif',
    'bouguer':      'onshore_geodetic_Spherical_Cap_Bouguer_2016_hsi_black.tif',
    'isostatic':    'onshore_geodetic_Isostatic_Residual_v2_2016_hsi_black.tif',
    'radiometrics': 'radmap_v4_2019_filtered_ternary_image.tif',
}

REE_OXIDE_COLS = ['ceo2_ppm','la2o3_ppm','nd2o3_ppm','pr6o11_ppm','sm2o3_ppm','eu2o3_ppm']
REE_ELEM_COLS  = ['ce_ppm','la_ppm','nd_ppm','pr_ppm','sm_ppm','eu_ppm']
OTHER_REE_COLS = ['gd_ppm','tb_ppm','dy_ppm','ho_ppm','er_ppm','yb_ppm','lu_ppm','y_ppm','sc_ppm','th_ppm','u_ppm']

log_lines = []
def log(msg): print(f"  {msg}"); log_lines.append(msg)

def find_file(filename, roots):
    for root in roots:
        for f in Path(root).rglob(filename):
            return f
    return None

def load_drillhole_data():
    import pandas as pd, numpy as np
    roots = [BASE_EXTRACTED, BASE_ORIGINAL]
    log("="*56); log("STEP 1: Loading drillhole data"); log("="*56)

    collar_path = find_file('dh_collar.csv', roots)
    if not collar_path: log("ERROR: dh_collar.csv not found!"); sys.exit(1)
    log(f"Collar: {collar_path}")
    collar = pd.read_csv(collar_path, low_memory=False)
    collar.columns = [c.strip().lower() for c in collar.columns]
    collar['lon'] = pd.to_numeric(collar.get('longitude', pd.Series(dtype=float)), errors='coerce')
    collar['lat'] = pd.to_numeric(collar.get('latitude',  pd.Series(dtype=float)), errors='coerce')
    collar['x']   = pd.to_numeric(collar.get('easting',   pd.Series(dtype=float)), errors='coerce')
    collar['y']   = pd.to_numeric(collar.get('northing',  pd.Series(dtype=float)), errors='coerce')
    collar['elevation'] = pd.to_numeric(collar.get('elevation', pd.Series(dtype=float)), errors='coerce')
    collar['maxdepth']  = pd.to_numeric(collar.get('maxdepth',  pd.Series(dtype=float)), errors='coerce')
    join_key = 'companyholeid' if 'companyholeid' in collar.columns else 'holeid'
    collar['join_key'] = collar[join_key].astype(str).str.strip()
    log(f"Collar rows: {len(collar):,}  join key: {join_key}")

    assay_path = find_file('dh_assay_pivoted.csv', roots)
    if not assay_path: log("ERROR: dh_assay_pivoted.csv not found!"); sys.exit(1)
    log(f"Assay: {assay_path}")
    assay = pd.read_csv(assay_path, low_memory=False)
    assay.columns = [c.strip().lower() for c in assay.columns]

    ree_oxide_p = [c for c in REE_OXIDE_COLS if c in assay.columns]
    ree_elem_p  = [c for c in REE_ELEM_COLS  if c in assay.columns]
    other_p     = [c for c in OTHER_REE_COLS  if c in assay.columns]
    all_ree = ree_oxide_p + ree_elem_p + other_p
    log(f"REE oxide cols: {ree_oxide_p}")
    log(f"REE element cols: {ree_elem_p}")

    for c in all_ree:
        assay[c] = pd.to_numeric(assay[c], errors='coerce')
    assay['fromdepth'] = pd.to_numeric(assay.get('fromdepth', pd.Series(dtype=float)), errors='coerce')
    assay['todepth']   = pd.to_numeric(assay.get('todepth',   pd.Series(dtype=float)), errors='coerce')

    treo_cols = ree_oxide_p if ree_oxide_p else ree_elem_p
    assay['treo_interval'] = assay[treo_cols].sum(axis=1, skipna=True)
    assay.loc[assay[treo_cols].isna().all(axis=1), 'treo_interval'] = np.nan

    lree_c = [c for c in ['ceo2_ppm','la2o3_ppm','nd2o3_ppm','pr6o11_ppm','sm2o3_ppm','ce_ppm','la_ppm','nd_ppm'] if c in assay.columns]
    hree_c = [c for c in ['gd_ppm','tb_ppm','dy_ppm','ho_ppm','er_ppm','yb_ppm','lu_ppm','y_ppm'] if c in assay.columns]
    if lree_c: assay['lree'] = assay[lree_c].sum(axis=1, skipna=True)
    if hree_c: assay['hree'] = assay[hree_c].sum(axis=1, skipna=True)
    if lree_c and hree_c: assay['lree_hree_ratio'] = assay['lree'] / (assay['hree'] + 0.001)

    jk_assay = 'companyholeid' if 'companyholeid' in assay.columns else 'anumber'
    assay['join_key'] = assay[jk_assay].astype(str).str.strip()

    agg_d = {c: ['max','mean'] for c in all_ree + ['treo_interval']}
    if 'lree' in assay.columns: agg_d['lree'] = ['max','mean']
    if 'hree' in assay.columns: agg_d['hree'] = ['max','mean']
    if 'lree_hree_ratio' in assay.columns: agg_d['lree_hree_ratio'] = ['mean']
    agg_d['fromdepth'] = 'min'
    agg_d['todepth']   = 'max'
    assay_agg = assay.groupby('join_key').agg(agg_d)
    assay_agg.columns = ['_'.join(c).strip('_') for c in assay_agg.columns]
    assay_agg = assay_agg.reset_index()
    log(f"Aggregated to {len(assay_agg):,} drillholes  TREO max={assay['treo_interval'].max():.1f} mean={assay['treo_interval'].mean():.2f}")

    alter_path = find_file('dh_alteration.csv', roots)
    alter_agg = None
    if alter_path:
        alter = pd.read_csv(alter_path, low_memory=False)
        alter.columns = [c.strip().lower() for c in alter.columns]
        if 'collarid' in alter.columns:
            alter['join_key'] = alter['collarid'].astype(str).str.strip()
            alter['altval'] = alter.get('attributevalue', pd.Series(dtype=str)).astype(str).str.lower()
            for kw in ['laterit','carbonat','weath','oxid','saprolite','clay','goethit','limonit']:
                alter[f'alt_{kw}'] = alter['altval'].str.contains(kw, na=False).astype(int)
            alt_cols = [c for c in alter.columns if c.startswith('alt_')]
            alter_agg = alter.groupby('join_key')[alt_cols].max().reset_index()
            log(f"Alteration: {alt_cols}")

    return collar, assay_agg, alter_agg, join_key

def build_master(collar, assay_agg, alter_agg):
    import pandas as pd, numpy as np
    log("\n"+"="*56); log("STEP 2: Merging tables"); log("="*56)
    master = collar.merge(assay_agg, on='join_key', how='left')
    matched = master['treo_interval_max'].notna().sum()
    log(f"Matched drillholes: {matched:,} / {len(master):,}")
    if alter_agg is not None:
        master = master.merge(alter_agg, on='join_key', how='left')
    def ds(d):
        if pd.isna(d): return 0.5
        if d<20: return 1.0
        if d<50: return 0.95
        if d<80: return 0.85
        if d<150: return 0.65
        if d<300: return 0.45
        return 0.25
    master['depth_score'] = master['fromdepth_min'].apply(ds)
    treo = master['treo_interval_max'].dropna()
    p95 = treo.quantile(0.95) if len(treo) > 0 and treo.quantile(0.95) > 0 else 1.0
    master['treo_norm'] = (master['treo_interval_max'] / p95).clip(0,1).fillna(0)
    return master

def extract_rasters(master):
    import numpy as np
    log("\n"+"="*56); log("STEP 3: Extracting geophysics at drillhole locations"); log("="*56)
    try:
        import rasterio
    except ImportError:
        log("rasterio not installed -- skipping (run: pip install rasterio)")
        for k in RASTER_TARGETS: master[f'geophys_{k}'] = 0.0
        return master

    if 'lat' not in master.columns or master['lat'].notna().sum() == 0:
        try:
            from pyproj import Transformer
            tr = Transformer.from_crs("EPSG:28351","EPSG:4326",always_xy=True)
            master['lon'], master['lat'] = tr.transform(master['x'].values, master['y'].values)
            log("Reprojected MGA51 -> WGS84")
        except:
            log("Cannot reproject -- skipping raster extraction")
            for k in RASTER_TARGETS: master[f'geophys_{k}'] = 0.0
            return master

    for key, fname in RASTER_TARGETS.items():
        rpath = None
        for root in [BASE_EXTRACTED, BASE_ORIGINAL]:
            for f in Path(root).rglob(fname):
                rpath = f; break
            if rpath: break
        if not rpath:
            log(f"[SKIP] {key}: {fname} not found"); master[f'geophys_{key}'] = 0.0; continue
        log(f"Extracting {key} <- {rpath.name}")
        try:
            with rasterio.open(rpath) as src:
                coords = list(zip(master['lon'].fillna(0), master['lat'].fillna(0)))
                vals = np.array([v[0] for v in src.sample(coords)], dtype=float)
                if src.nodata is not None: vals[vals == src.nodata] = np.nan
                master[f'geophys_{key}'] = vals
                log(f"  valid={np.isfinite(vals).sum():,}  mean={np.nanmean(vals):.2f}")
        except Exception as e:
            log(f"  ERROR: {e}"); master[f'geophys_{key}'] = 0.0
    return master

def join_geology(master):
    log("\n"+"="*56); log("STEP 4: Spatial join with geology"); log("="*56)
    try:
        import geopandas as gpd
        from shapely.geometry import Point
    except ImportError:
        log("geopandas not installed -- skipping (run: pip install geopandas)")
        return master

    use_latlon = 'lat' in master.columns and master['lat'].notna().any()
    geom = [Point(row['lon'] if use_latlon else row.get('x',0),
                  row['lat'] if use_latlon else row.get('y',0)) for _, row in master.iterrows()]
    pts = gpd.GeoDataFrame(master, geometry=geom, crs='EPSG:4326' if use_latlon else 'EPSG:28351')

    for shpname, prefix in [('GeologicUnitPolygons2_5M.shp','geol'),('ProvinceFullExtent.shp','prov')]:
        sp = None
        for root in [BASE_EXTRACTED, BASE_ORIGINAL]:
            for f in Path(root).rglob(shpname):
                sp = f; break
            if sp: break
        if not sp: log(f"[SKIP] {shpname} not found"); continue
        log(f"Joining {shpname}")
        try:
            gdf = gpd.read_file(sp)
            pts2 = pts.to_crs(gdf.crs) if pts.crs != gdf.crs else pts
            name_cols = [c for c in gdf.columns if any(k in c.lower() for k in ['name','unit','lith','type','age']) and c!='geometry'][:4]
            joined = gpd.sjoin(pts2, gdf[['geometry']+name_cols], how='left', predicate='within')
            for c in name_cols:
                if c in joined.columns: master[f'{prefix}_{c}'] = joined[c].values
            log(f"  Added: {[prefix+'_'+c for c in name_cols]}")
        except Exception as e:
            log(f"  ERROR: {e}")
    return master

def finalise(master):
    import pandas as pd
    log("\n"+"="*56); log("STEP 5: Building ML feature matrix"); log("="*56)
    ree_feat  = [c for c in master.columns if any(k in c for k in ['treo','ceo2','la2o3','nd2o3','pr6','sm2','eu2','lree','hree','ratio','_ppm_max','_ppm_mean'])]
    gphy_feat = [c for c in master.columns if c.startswith('geophys_')]
    depth_feat= [c for c in ['depth_score','treo_norm','maxdepth','fromdepth_min'] if c in master.columns]
    alt_feat  = [c for c in master.columns if c.startswith('alt_')]
    all_feat  = list(dict.fromkeys(ree_feat+gphy_feat+depth_feat+alt_feat))
    log(f"Features: {len(ree_feat)} REE + {len(gphy_feat)} geophys + {len(depth_feat)} depth + {len(alt_feat)} alteration = {len(all_feat)} total")

    geol_cat = [c for c in master.columns if c.startswith('geol_') or c.startswith('prov_')]
    for c in geol_cat:
        if master[c].dtype == object:
            d = pd.get_dummies(master[c].fillna('unknown'), prefix=c, drop_first=True)
            master = pd.concat([master, d], axis=1)
            all_feat += list(d.columns)

    for c in all_feat:
        if c in master.columns and master[c].dtype in ['float64','int64',float,int]:
            master[c] = master[c].fillna(master[c].median())

    spa = [c for c in ['join_key','lat','lon','x','y','elevation','holeid','companyholeid'] if c in master.columns]
    out = spa + [c for c in all_feat if c in master.columns]
    out = list(dict.fromkeys(out))
    return master[out], all_feat

if __name__ == "__main__":
    print("\n" + "=" * 56)
    print("  REE PROSPECTIVITY ENGINE -- DATA PROCESSING")
    print("=" * 56)
    log(f"Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")

    import pandas as pd
    import numpy as np

    out = Path(OUTPUT_FOLDER)
    out.mkdir(parents=True, exist_ok=True)

    collar, assay_agg, alter_agg, jk = load_drillhole_data()
    master = build_master(collar, assay_agg, alter_agg)
    master = extract_rasters(master)
    master = join_geology(master)
    fm, feat_cols = finalise(master)

    fm.to_csv(out / "feature_matrix.csv", index=False, encoding="utf-8")
    master.to_csv(out / "master_drillholes.csv", index=False, encoding="utf-8")
    (out / "feature_list.txt").write_text("\n".join(feat_cols), encoding="utf-8")
    (out / "processing_log.txt").write_text("\n".join(log_lines), encoding="utf-8")

    log(f"\nFeature matrix: {fm.shape[0]:,} rows x {fm.shape[1]} cols")
    print("\n  SUCCESS! Run step3_map.py next.\n")

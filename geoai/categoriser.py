"""
geoai/categoriser.py
Auto-categorises any uploaded file into a geoscience data layer.
No configuration needed — works by filename, extension, and column headers.
"""
from pathlib import Path
import re

# ── LAYER DEFINITIONS ─────────────────────────────────────────
LAYERS = ["geophysics", "geochemical", "geology", "satellite", "topography", "drillhole"]

# Keywords that appear in filenames
FILENAME_RULES = {
    "geophysics":  ["tmi","mag","magneti","gravity","bouguer","isostatic","radiometri",
                    "gammaray","gamma_ray","aeromagnet","gravimetri","gravi","emag",
                    "resistiv","ip_","induced","susceptib","conductiv"],
    "geochemical": ["geochem","stream_sed","soil_","lithogeochem","regolith"],
    "drillhole":   ["collar","survey","litholog","dh_","drill","hole","interval","downhole",
                    "from_to","assay","assay_pivot","lith","alteration","structure",
                    "element","ppm","ppb","sample"],
    "geology":     ["geol","lithol","formation","unit","contact","fault","fold","province",
                    "stratigraphy","map","outcrop","rock","struct"],
    "satellite":   ["sentinel","landsat","aster","modis","band","reflec","ndvi","swir",
                    "l2a","l2sp","surface_ref","satellite","imagery","msil","l1tp"],
    "topography":  ["dem","dtm","dsm","elevation","srtm","lidar","bathymetry","relief",
                    "slope","aspect","topograph","hillshade","terrain"],
}

# File extension rules
EXTENSION_RULES = {
    "geophysics":  [".ers",".grd",".dat",".xyz",".gxf",".bil",".hdr",".nc",".h5",".hdf"],
    "drillhole":   [".csv",".xlsx",".xls",".txt",".accdb",".mdb"],
    "geology":     [".shp",".gpkg",".geojson",".kml",".kmz",".gdb",".e00",".dxf"],
    "satellite":   [".jp2",".ecw",".img"],
    "topography":  [".asc",".laz",".las",".xyz"],
    "geophysics":  [".tif",".tiff"],   # default for rasters
    "archive":     [".zip", ".tar", ".gz", ".7z", ".rar"],
}

# Column header keywords (for CSV/Excel)
COLUMN_RULES = {
    "drillhole":   ["holeid","companyholeid","collar","fromm","from_depth","fromdepth",
                    "anumber","dip","azimuth","maxdepth","easting","northing",
                    "ceo2","la2o3","nd2o3","treo","lree","hree","ppm","ppb"],
    "geochemical": ["au","ag","cu","pb","zn","mo","as","sb","bi","w","sn","element",
                    "result","value","units","detection","lab","sample_id"],
    "geology":     ["lithology","formation","unit","description","rock_type","lith_code",
                    "stratigraph","age","period","epoch"],
    "geophysics":  ["tmi","magnetic","gravity","radiometric","uranium","thorium","potassium",
                    "bouguer","isostatic","apparent_resistivity"],
    "topography":  ["elevation","height","z","altitude","dem","dtm","slope","aspect"],
}

def categorise_file(filepath):
    """
    Determine the geoscience layer for a file.
    Returns dict with: layer, confidence, reasons
    """
    p      = Path(str(filepath))
    name   = p.stem.lower().replace("-","_").replace(" ","_")
    ext    = p.suffix.lower()
    scores = {layer: 0 for layer in LAYERS}
    reasons = []

    # ── Score by filename keywords ────────────────────────────
    for layer, keywords in FILENAME_RULES.items():
        for kw in keywords:
            if kw in name:
                weight = 10 if kw in ["collar", "assay", "dh_", "drillhole"] else 3
                scores[layer] += weight
                reasons.append(f"filename contains '{kw}' → {layer}")
                break

    # ── Score by extension ────────────────────────────────────
    for layer, exts in EXTENSION_RULES.items():
        if ext in exts:
            scores[layer] += 2
            reasons.append(f"extension '{ext}' → {layer}")
            break

    # ── Score by column headers (CSV/Excel only) ──────────────
    if ext in [".csv",".txt",".xlsx",".xls"]:
        try:
            if ext == ".csv":
                import pandas as pd
                cols = pd.read_csv(str(p), nrows=0).columns.tolist()
            else:
                import pandas as pd
                cols = pd.read_excel(str(p), nrows=0).columns.tolist()
            cols_lower = " ".join(str(c).lower() for c in cols)
            for layer, keywords in COLUMN_RULES.items():
                hits = [kw for kw in keywords if kw in cols_lower]
                if hits:
                    scores[layer] += len(hits) * 2
                    reasons.append(f"columns {hits[:3]} → {layer}")
        except Exception:
            pass

    # ── Special: .tif with satellite keywords → satellite ─────
    if ext in [".tif",".tiff"]:
        sat_kws = FILENAME_RULES["satellite"]
        if any(k in name for k in sat_kws):
            scores["satellite"] += 5
        else:
            scores["geophysics"] += 2   # default raster = geophysics

    best  = max(scores, key=scores.get)
    total = sum(scores.values())
    conf  = round(scores[best] / max(total, 1), 2)

    return {
        "layer":      best,
        "confidence": conf,
        "scores":     scores,
        "reasons":    reasons[:4],
        "filename":   p.name,
        "extension":  ext,
    }

def categorise_batch(filepaths):
    """Categorise a list of files and return grouped summary."""
    results = []
    groups  = {layer: [] for layer in LAYERS}
    for fp in filepaths:
        r = categorise_file(fp)
        results.append(r)
        groups[r["layer"]].append(fp)
    return results, groups

def detect_deposit_name(filepaths):
    """
    Try to infer deposit name from folder structure or filenames.
    Returns best guess string.
    """
    known = {
        "mount_weld":     ["mount_weld","mtweld","mw_","weld","mwgc"],
        "mountain_pass":  ["mountain_pass","mtpass","moly","mp_"],
        "bayan_obo":      ["bayan","baiyun","baotou"],
        "browns_range":   ["browns","brown_range","br_","dysprosium"],
        "ngualla":        ["ngualla","peak_rare"],
        "kvanefjeld":     ["kvane","greenland"],
        "lynas":          ["lynas","cld_","central_lanthan"],
    }
    names = " ".join(Path(str(f)).stem.lower() for f in filepaths)
    for deposit, keywords in known.items():
        if any(k in names for k in keywords):
            return deposit
    # Fall back to most common folder name
    folders = [Path(str(f)).parent.name.lower().replace(" ","_")
               for f in filepaths]
    if folders:
        from collections import Counter
        return Counter(folders).most_common(1)[0][0]
    return "unknown_deposit"

"""
REE ENGINE -- STEP 1: FULL DATA INVENTORY
Scans ALL folders and produces a detailed report.
"""
import os, sys, json, datetime
from pathlib import Path
from collections import defaultdict

# ── ALL folders to scan ────────────────────────────────
DATA_FOLDERS = [
    r"D:\GeoAI-INDIA\training_data",
    r"D:\GeoAI-INDIA\training_data_extracted",
]
REPORT_FOLDER = r"D:\GeoAI-INDIA"
# ──────────────────────────────────────────────────────

def human_size(b):
    for u in ['B','KB','MB','GB']:
        if b < 1024: return f"{b:.1f} {u}"
        b /= 1024
    return f"{b:.1f} TB"

def categorise(ext):
    cats = {
        'tabular':  ['.csv','.txt','.tsv','.xlsx','.xls'],
        'gis':      ['.shp','.gpkg','.geojson','.kml','.gdb'],
        'raster':   ['.tif','.tiff','.asc','.grd','.ers','.img','.nc','.hdf','.h5'],
        'database': ['.db','.sqlite','.mdb'],
        'document': ['.pdf','.doc','.docx'],
        'image':    ['.jpg','.jpeg','.png','.ecw','.jp2'],
        'geophys':  ['.dat','.xyz','.gxf'],
        'data':     ['.json','.npy','.pkl','.parquet'],
        'archive':  ['.tar','.zip','.gz','.bz2','.tgz'],
    }
    for cat, exts in cats.items():
        if ext in exts: return cat
    return 'other'

REE_KW = ['ce','la','nd','pr','sm','eu','treo','tree','ree','lree','hree',
           'ceo2','la2o3','rare earth','rare_earth']

lines = []
def pr(s): print(f"  {s}"); lines.append(s)

print("\n" + "="*62)
print("  REE ENGINE -- DATA INVENTORY (multi-folder)")
print("="*62 + "\n")

grand = defaultdict(lambda:{'count':0,'size':0,'exts':set()})
all_paths = defaultdict(list)
all_ree   = []
all_cols  = {}
all_json  = []
folder_summaries = []

for folder in DATA_FOLDERS:
    p = Path(folder)
    if not p.exists():
        print(f"  [SKIP] Not found: {folder}")
        continue

    print(f"  Scanning: {folder} ...")
    f_count = 0; f_size = 0
    ext_groups = defaultdict(list)

    for f in p.rglob('*'):
        if not f.is_file(): continue
        size = f.stat().st_size
        f_count += 1; f_size += size
        ext = f.suffix.lower()
        ext_groups[ext].append({'name':f.name,'path':str(f),'size':size})

    folder_summaries.append((folder, f_count, f_size))

    for ext, files in ext_groups.items():
        cat = categorise(ext)
        grand[cat]['count'] += len(files)
        grand[cat]['size']  += sum(fi['size'] for fi in files)
        grand[cat]['exts'].add(ext or '(none)')
        all_paths[ext] += [fi['path'] for fi in files]

    # Peek inside CSVs
    for info in ext_groups.get('.csv',[])[:15]:
        try:
            with open(info['path'],'r',errors='ignore') as fh:
                hdr = fh.readline().strip().lower()
                cols = [c.strip().strip('"') for c in hdr.split(',')][:25]
                all_cols[info['name']] = cols
                for col in cols:
                    if any(k in col for k in REE_KW):
                        all_ree.append(f"{info['name']} -> column: {col}")
        except: pass

    # Peek inside JSONs
    for info in ext_groups.get('.json',[])[:5]:
        try:
            import json as _j
            with open(info['path'],'r',errors='ignore') as fh:
                obj = _j.load(fh)
                if isinstance(obj,dict):
                    all_json.append(f"{info['name']}: keys={list(obj.keys())[:6]}")
                elif isinstance(obj,list) and obj:
                    all_json.append(f"{info['name']}: list[{len(obj)}] first={list(obj[0].keys())[:5] if isinstance(obj[0],dict) else type(obj[0]).__name__}")
        except: pass

# ── Build report ─────────────────────────────────────
sep = "=" * 62
pr(sep)
pr("  REE PROSPECTIVITY ENGINE -- DATA INVENTORY REPORT")
pr(f"  Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
pr(sep); pr("")

pr("FOLDERS SCANNED")
pr("---------------")
grand_files = grand_size = 0
for folder, fc, fs in folder_summaries:
    pr(f"  {folder}")
    pr(f"    Files: {fc:,}   Size: {human_size(fs)}")
    grand_files += fc; grand_size += fs
pr("")
pr(f"COMBINED TOTALS: {grand_files:,} files   {human_size(grand_size)}")
pr("")

pr("FILE TYPES BY CATEGORY")
pr("----------------------")
priority = ['tabular','data','raster','gis','geophys','image','database','document','archive','other']
for cat in priority:
    if cat in grand:
        info = grand[cat]
        exts = ', '.join(sorted(info['exts']))
        pr(f"  {cat.upper():<12}  {info['count']:>6} files  {human_size(info['size']):>9}   [{exts}]")
pr("")

# Flag unextracted archives
arc_paths = all_paths.get('.tar',[]) + all_paths.get('.zip',[]) + \
            all_paths.get('.gz',[])  + all_paths.get('.tgz',[])
if arc_paths:
    pr("UNEXTRACTED ARCHIVES (run step0 to extract these)")
    pr("--------------------------------------------------")
    for ap in arc_paths[:10]:
        sz = human_size(Path(ap).stat().st_size)
        pr(f"  {sz:>9}  {ap}")
    if len(arc_paths) > 10:
        pr(f"  ... and {len(arc_paths)-10} more")
    pr("")

if all_ree:
    pr("REE / GEOCHEMICAL COLUMNS DETECTED")
    pr("-----------------------------------")
    for h in all_ree[:40]: pr(f"  [FOUND] {h}")
    pr("")

if all_json:
    pr("JSON FILE STRUCTURE")
    pr("-------------------")
    for j in all_json[:8]: pr(f"  {j}")
    pr("")

if all_cols:
    pr("CSV COLUMN HEADERS (sample)")
    pr("---------------------------")
    for fname, cols in list(all_cols.items())[:12]:
        non_empty = [c for c in cols if c.strip()]
        pr(f"  {fname}")
        pr(f"    {', '.join(non_empty[:20])}{'...' if len(non_empty)>20 else ''}")
    pr("")

# Detailed file listings for ML-useful types
for ext, label in [('.tif','RASTER (GeoTIFF)'),('.ers','RASTER (ERS)'),
                   ('.img','RASTER (IMG)'), ('.shp','GIS (Shapefile)'),
                   ('.nc','RASTER (NetCDF)'),('.h5','DATA (HDF5)'),
                   ('.npy','DATA (NumPy)'),('.db','DATABASE')]:
    paths = all_paths.get(ext,[])
    if paths:
        pr(f"{label} -- {len(paths)} files")
        for ap in sorted(paths)[:8]: pr(f"  {ap}")
        if len(paths)>8: pr(f"  ... and {len(paths)-8} more")
        pr("")

pr(sep)
pr("  Upload this file to Claude for customised next steps.")
pr(sep)

report_txt = '\n'.join(lines)
# Save report
for loc in [REPORT_FOLDER, str(Path(__file__).parent)]:
    try:
        rp = Path(loc) / 'inventory_report.txt'
        rp.parent.mkdir(parents=True, exist_ok=True)
        rp.write_text(report_txt, encoding='utf-8')
        print(f"\n  Report saved: {rp}")
        break
    except: pass

# Save JSON index
try:
    idx = Path(__file__).parent / 'inventory_data.json'
    idx.write_text(json.dumps({
        'ext_paths': {k: v[:30] for k,v in all_paths.items()},
        'csv_cols':  all_cols,
        'ree_hints': all_ree,
    }, indent=2), encoding='utf-8')
    print(f"  Data index: {idx}")
except: pass

print("\n  Done! Upload inventory_report.txt to Claude.\n")
input("  Press Enter to close...")

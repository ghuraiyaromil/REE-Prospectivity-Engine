"""
GeoAI — REE Prospectivity Engine
Upload any raw deposit data → get a prospectivity map.
"""
import streamlit as st, pandas as pd, numpy as np
import folium, tempfile, shutil, json
from pathlib import Path
from streamlit_folium import st_folium
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

st.set_page_config(page_title="GeoAI — REE Prospectivity",
                   page_icon="🪨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif}
h1{font-family:'IBM Plex Mono',monospace!important;color:#C9A84C!important;font-size:1.8rem!important}
h3{color:#C9A84C!important}
.stButton>button{background:#C9A84C;color:#0d0a06;border:none;font-family:'IBM Plex Mono',monospace;
  font-weight:500;border-radius:4px;padding:.6rem 2rem;font-size:1rem;width:100%}
.stButton>button:hover{background:#E8D5A0}
.mbox{background:#1a1208;border:1px solid #3D2E14;border-radius:8px;padding:1rem;text-align:center}
.mval{font-family:'IBM Plex Mono',monospace;font-size:1.6rem;font-weight:500;color:#C9A84C}
.mlbl{font-size:.7rem;color:#6B5535;text-transform:uppercase;letter-spacing:1px;margin-top:4px}
.logbox{background:#0d0a06;border:1px solid #3D2E14;border-radius:6px;padding:1rem;
  font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:#6B8C6B;max-height:220px;overflow-y:auto}
.pill{display:inline-block;background:#1a1208;border:1px solid #3D2E14;border-radius:4px;
  padding:3px 10px;margin:2px;font-family:'IBM Plex Mono',monospace;font-size:.75rem}
.badge{display:inline-block;border-radius:3px;padding:2px 8px;font-size:.7rem;margin:2px;
  font-family:'IBM Plex Mono',monospace}
div[data-testid="stSidebar"]{background:#080604;border-right:1px solid #1a1208}
</style>
""", unsafe_allow_html=True)

LAYER_COL = {
    "geophysics":("#1D9E75","#E1F5EE"),  "geochemical":("#BA7517","#FAEEDA"),
    "drillhole": ("#378ADD","#E6F1FB"),  "geology":    ("#7F77DD","#EEEDFE"),
    "satellite": ("#D85A30","#FAECE7"),  "topography": ("#888780","#F1EFE8"),
}
OUTPUT_DIR = r"D:\GeoAI-INDIA\ree_output"

# ── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪨 GeoAI")
    st.markdown("**REE Prospectivity Engine**")
    st.markdown("---")
    st.markdown("**Accepts any format**")
    for line in ["Drillhole: CSV, XLS, XLSX","Geophysics: TIF, ERS, GRD",
                 "Geology: SHP, GeoJSON","Satellite: JP2, ECW","Topography: ASC, XYZ",
                 "Archives: ZIP, TAR"]:
        st.markdown(f"- {line}")
    st.markdown("---")
    try:
        reg_path = Path(OUTPUT_DIR)/"deposit_registry.json"
        if reg_path.exists():
            reg = json.loads(reg_path.read_text())
            deposits = reg.get("deposits",{})
            if deposits:
                st.markdown("**Trained deposits:**")
                for d,info in deposits.items():
                    v = info["versions"][-1]
                    bg,fg = LAYER_COL["geophysics"]
                    st.markdown(f'<span class="badge" style="background:{bg};color:{fg}">'
                                f'{d}</span> R²={v["cv_r2"]}', unsafe_allow_html=True)
    except Exception:
        pass

# ── HEADER ───────────────────────────────────────────────────
st.markdown("# GeoAI — REE Prospectivity Engine")
st.markdown("Upload raw exploration data from any REE deposit. "
            "The pipeline auto-categorises, cleans, extracts features, trains and scores.")
st.markdown("---")

st.markdown("### Upload deposit data")
c1, c2 = st.columns([3,1])
with c1:
    uploaded = st.file_uploader(
        "Any files — CSV, TIF, SHP, ZIP, XLS, JP2, or anything else",
        accept_multiple_files=True, type=None, label_visibility="collapsed")
with c2:
    dep_name = st.text_input("Deposit name (optional)",
                             placeholder="e.g. mount_weld")

if uploaded:
    from geoai.categoriser import categorise_file
    cats, tmp_paths = [], []
    for f in uploaded:
        suf = Path(f.name).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
        tmp.write(f.read()); tmp.close()
        r = categorise_file(tmp.name)
        r["tmp_path"] = tmp.name; r["original_name"] = f.name
        cats.append(r); tmp_paths.append(tmp.name)

    st.markdown(f"**{len(uploaded)} file(s) uploaded**")
    cols = st.columns(min(len(cats),4))
    for i,r in enumerate(cats):
        bg,fg = LAYER_COL.get(r["layer"],("#333","#eee"))
        with cols[i%4]:
            st.markdown(f'<div class="pill">{r["original_name"][:22]}<br>'
                        f'<span class="badge" style="background:{bg};color:{fg}">'
                        f'{r["layer"]}</span></div>', unsafe_allow_html=True)

    from collections import Counter
    lc = Counter(r["layer"] for r in cats)
    html = ""
    for layer,count in lc.items():
        bg,fg = LAYER_COL.get(layer,("#333","#eee"))
        html += f'<span class="badge" style="background:{bg};color:{fg}">{layer}: {count}</span>'
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("")

    if st.button("Run prospectivity analysis"):
        log_lines = []; log_ph = st.empty()

        def upd_log(msg):
            log_lines.append(f"› {msg}")
            log_ph.markdown(
                '<div class="logbox">'+"<br>".join(log_lines[-18:])+"</div>",
                unsafe_allow_html=True)

        tmp_dir = Path(tempfile.mkdtemp())
        fps = []
        for r in cats:
            dst = tmp_dir/r["original_name"]
            shutil.copy(r["tmp_path"], str(dst)); fps.append(dst)

        try:
            from geoai.pipeline import GeoAIPipeline
            pipe   = GeoAIPipeline(output_dir=OUTPUT_DIR)
            result = pipe.run(files=fps,
                              deposit_name=dep_name.strip() or None,
                              progress_cb=upd_log)
        except Exception as e:
            import traceback
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            result = {"status":"error"}
        finally:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        if result.get("status") == "success":
            st.success("Analysis complete!")
            master   = result["master_df"]
            treo_col = result["treo_col"]

            # Metrics
            m = result
            st.markdown(
                f'<div style="display:flex;gap:1rem;margin:1rem 0">'
                f'<div class="mbox"><div class="mval">{m["cv_r2"]:.3f}</div>'
                f'<div class="mlbl">CV R²</div></div>'
                f'<div class="mbox"><div class="mval">{m["roc_auc"]:.3f}</div>'
                f'<div class="mlbl">ROC AUC</div></div>'
                f'<div class="mbox"><div class="mval">{m["n_labelled"]}</div>'
                f'<div class="mlbl">Labelled holes</div></div>'
                f'<div class="mbox"><div class="mval">{m["top_score"]:.0f}</div>'
                f'<div class="mlbl">Top score</div></div>'
                f'<div class="mbox"><div class="mval">{m["n_features"]}</div>'
                f'<div class="mlbl">Features</div></div></div>',
                unsafe_allow_html=True)

            # Map
            if "lat" in master.columns and "lon" in master.columns:
                st.markdown("#### Prospectivity map")
                valid = master.dropna(subset=["lat","lon"])
                valid = valid[valid["lat"].abs() < 90]
                if len(valid):
                    mp = folium.Map([float(valid["lat"].median()),
                                     float(valid["lon"].median())],
                                    zoom_start=13, tiles="CartoDB dark_matter")
                    from folium.plugins import HeatMap, MarkerCluster
                    heat = [[float(r.lat),float(r.lon),float(r.score_100)/100]
                            for r in valid.itertuples()
                            if np.isfinite(r.lat) and np.isfinite(r.lon)]
                    if heat:
                        HeatMap(heat, radius=16, blur=12,
                                gradient={0:"#1a0a00",0.4:"#5C2010",0.65:"#A0400A",
                                          0.8:"#D4601A",0.92:"#C9A84C",1:"#FFF5D0"},
                                min_opacity=0.4).add_to(mp)
                    mc = MarkerCluster().add_to(mp)
                    for _,row in valid.nlargest(30,"score_100").iterrows():
                        sc = float(row["score_100"])
                        cl = "#C9A84C" if sc>=75 else "#D4601A" if sc>=60 else "#8B4513"
                        tv = (f"{row[treo_col]:,.0f} ppm"
                              if treo_col and pd.notna(row.get(treo_col)) else "predicted")
                        folium.CircleMarker(
                            [float(row["lat"]),float(row["lon"])],
                            radius=9 if sc>=75 else 6, color=cl,
                            fill=True, fill_opacity=0.85,
                            popup=folium.Popup(
                                f"<b>{row.get('companyholeid','')}</b><br>"
                                f"Score: <b>{sc:.0f}/100</b><br>TREO: {tv}",max_width=200),
                            tooltip=f"{row.get('companyholeid','')} {sc:.0f}/100"
                        ).add_to(mc)
                    folium.LayerControl().add_to(mp)
                    st_folium(mp, width=None, height=500)

            # Table
            st.markdown("#### Top 20 drill targets")
            show = [c for c in ["companyholeid","lat","lon","score_100",treo_col,"depth_score"]
                    if c and c in master.columns]
            st.dataframe(master.nlargest(20,"score_100")[show], use_container_width=True)

            # Model scores
            if result.get("model_scores"):
                st.markdown("#### Model comparison")
                ms = dict(result["model_scores"]); ms["Ensemble"] = m["cv_r2"]
                for n,r2 in sorted(ms.items(),key=lambda x:-x[1]):
                    st.markdown(f'`{n.upper():<10}` **{r2:+.4f}** {"█"*max(0,int(r2*30))}')

            # Download
            st.download_button("Download full results (CSV)",
                               master.to_csv(index=False).encode(),
                               f"ree_results_{m['deposit']}.csv","text/csv")

        elif result.get("status") == "insufficient_data":
            st.warning(f"Only {result['n_labelled']} labelled samples found. "
                       "Upload assay data with TREO grades to enable training.")

else:
    st.markdown("""
<div style="border:1px dashed #3D2E14;border-radius:8px;padding:3rem;
            text-align:center;color:#6B5535;margin:2rem 0">
  <div style="font-size:2rem;margin-bottom:1rem">🪨</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;margin-bottom:.5rem">
    Upload exploration data above to begin
  </div>
  <div style="font-size:.8rem">CSV · TIF · SHP · JP2 · XLS · ZIP · TAR · any format</div>
</div>""", unsafe_allow_html=True)
    st.markdown("#### What to upload")
    st.dataframe(pd.DataFrame({
        "File type": ["Collar CSV","Assay CSV","Geophysics TIF","Geology SHP","Satellite JP2"],
        "Auto-detected as": ["drillhole","drillhole","geophysics","geology","satellite"],
        "Required?": ["Yes","Recommended","Optional","Optional","Optional"],
    }), use_container_width=True, hide_index=True)

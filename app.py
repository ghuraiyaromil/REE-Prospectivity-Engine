"""
Geo-AI-India — Mineral Exploration AI
Upload any raw deposit data → get a prospectivity map.
"""
import json
import tempfile
import shutil
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
import matplotlib
matplotlib.use("Agg")

from streamlit_folium import st_folium
from fpdf import FPDF
from geoai.config import OUTPUT_DIR, VERSION

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Geo-AI-India — REE Prospectivity",
    page_icon="⛏️",
    layout="wide",
)

# ── THEME CSS ────────────────────────────────────────────────
_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --primary: #D4AF37;
    --bg-dark: #0A0A0B;
    --glass: rgba(255, 255, 255, 0.05);
    --glass-border: rgba(255, 255, 255, 0.1);
    --accent-orange: #FF6B2B;
}

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #E0E0E0;
}

.stApp {
    background-color: var(--bg-dark);
    background-image: 
        radial-gradient(at 0% 0%, rgba(212, 175, 55, 0.05) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(255, 107, 43, 0.05) 0px, transparent 50%);
}

h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    background: linear-gradient(135deg, #FFF 0%, #D4AF37 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    letter-spacing: -1px;
}

h3 {
    color: var(--primary) !important;
    font-weight: 500 !important;
}

/* Glass Cards */
.glass-card {
    background: var(--glass);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}

.metric-card {
    background: rgba(212, 175, 55, 0.03);
    border: 1px solid rgba(212, 175, 55, 0.1);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    transition: transform 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    background: rgba(212, 175, 55, 0.06);
    border-color: rgba(212, 175, 55, 0.3);
}

.mval {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    color: var(--primary);
}

.mlbl {
    font-size: 0.75rem;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(135deg, #D4AF37 0%, #B8860B 100%);
    color: #000 !important;
    border: none;
    font-weight: 600;
    border-radius: 8px;
    padding: 0.8rem 2rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    width: 100%;
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2);
}

.stButton>button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 25px rgba(212, 175, 55, 0.4);
    background: linear-gradient(135deg, #E8C860 0%, #D4AF37 100%);
}

.logbox {
    background: #000;
    border: 1px solid var(--glass-border);
    border-radius: 8px;
    padding: 1rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #00FF41;
    max-height: 250px;
    overflow-y: auto;
    box-shadow: inset 0 0 10px rgba(0, 255, 65, 0.1);
}

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #050505;
    border-right: 1px solid var(--glass-border);
}

.badge {
    display: inline-block;
    border-radius: 4px;
    padding: 4px 10px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.5px;
}
</style>
"""
st.markdown(_CSS, unsafe_allow_html=True)

# ── CONSTANTS ────────────────────────────────────────────────
LAYER_COLOURS = {
    "geophysics":  ("#1D9E75", "#E1F5EE"),
    "geochemical": ("#BA7517", "#FAEEDA"),
    "drillhole":   ("#378ADD", "#E6F1FB"),
    "geology":     ("#7F77DD", "#EEEDFE"),
    "satellite":   ("#D85A30", "#FAECE7"),
    "topography":  ("#888780", "#F1EFE8"),
    "archive":     ("#444444", "#FFFFFF"),
}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def _render_metrics(result: dict) -> None:
    """Render the top-level KPI metrics row with premium glass cards."""
    cols = st.columns(5)
    metrics = [
        (f"{result['cv_r2']:.3f}", "CV R²"),
        (f"{result['roc_auc']:.3f}", "ROC AUC"),
        (f"{result['n_labelled']}", "Labelled Holes"),
        (f"{result['top_score']:.0f}/100", "Top Target Score"),
        (f"{result['n_features']}", "Active Features")
    ]
    
    for i, (val, lbl) in enumerate(metrics):
        with cols[i]:
            st.markdown(
                f'<div class="metric-card"><div class="mval">{val}</div>'
                f'<div class="mlbl">{lbl}</div></div>',
                unsafe_allow_html=True
            )


def _render_map(master: pd.DataFrame, treo_col: str | None) -> None:
    """Render the Folium prospectivity heatmap."""
    if "lat" not in master.columns or "lon" not in master.columns:
        return
    valid = master.dropna(subset=["lat", "lon"])
    valid = valid[valid["lat"].abs() < 90]
    if valid.empty:
        return

    mp = folium.Map(
        [float(valid["lat"].median()), float(valid["lon"].median())],
        zoom_start=13,
        tiles="CartoDB dark_matter",
    )
    from folium.plugins import HeatMap, MarkerCluster

    heat = [
        [float(r.lat), float(r.lon), float(r.score_100) / 100]
        for r in valid.itertuples()
        if np.isfinite(r.lat) and np.isfinite(r.lon)
    ]
    if heat:
        HeatMap(
            heat, radius=16, blur=12,
            gradient={0: "#1a0a00", 0.4: "#5C2010", 0.65: "#A0400A",
                      0.8: "#D4601A", 0.92: "#C9A84C", 1: "#FFF5D0"},
            min_opacity=0.4,
        ).add_to(mp)

    mc = MarkerCluster().add_to(mp)
    for _, row in valid.nlargest(30, "score_100").iterrows():
        sc = float(row["score_100"])
        cl = "#C9A84C" if sc >= 75 else "#D4601A" if sc >= 60 else "#8B4513"
        tv = (f"{row[treo_col]:,.0f} ppm"
              if treo_col and pd.notna(row.get(treo_col)) else "predicted")
        folium.CircleMarker(
            [float(row["lat"]), float(row["lon"])],
            radius=9 if sc >= 75 else 6, color=cl,
            fill=True, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{row.get('companyholeid', '')}</b><br>"
                f"Score: <b>{sc:.0f}/100</b><br>TREO: {tv}", max_width=200),
            tooltip=f"{row.get('companyholeid', '')} {sc:.0f}/100",
        ).add_to(mc)
    folium.LayerControl().add_to(mp)
    st_folium(mp, use_container_width=True, height=500)


def _render_data_health(master: pd.DataFrame, treo_col: str | None) -> None:
    """Render the Data Health / QA/QC dashboard."""
    with st.expander("🛠️ Data Health Dashboard (QA/QC)", expanded=False):
        st.markdown("#### Automated quality checks")
        qc_cols = st.columns(3)

        # 1. Missing data
        missing_pct = master.isna().mean().mean() * 100
        qc_cols[0].metric(
            "Data Completion", f"{100 - missing_pct:.1f}%",
            "Healthy" if missing_pct < 20 else "Action Required",
        )

        # 2. Coordinate validity
        if "lat" in master.columns:
            coord_outliers = len(master[(master["lat"] < -90) | (master["lat"] > 90)])
        else:
            coord_outliers = len(master)
        qc_cols[1].metric(
            "Spatial Integrity",
            f"{len(master) - coord_outliers}/{len(master)}",
            "Verified" if coord_outliers == 0 else "Fix Coordinates",
        )

        # 3. High-grade outliers
        if treo_col and treo_col in master.columns:
            q99 = master[treo_col].quantile(0.99)
            high_grade = int((master[treo_col] > q99).sum())
        else:
            high_grade = 0
        qc_cols[2].metric("Unusual Grades", str(high_grade),
                          "Outliers Flagged", delta_color="off")


def _render_3d_visualiser(master: pd.DataFrame, treo_col: str | None) -> None:
    """Render the 3D sub-surface explorer using Plotly."""
    if "elevation" not in master.columns:
        st.warning("3D visualization requires an 'elevation' column.")
        return
    depth_col = next(
        (c for c in master.columns if "fromdepth" in c), None
    )
    depth_vals = pd.to_numeric(
        master.get(depth_col, 0) if depth_col else 0, errors="coerce"
    ).fillna(0)
    master = master.copy()
    master["depth_z"] = pd.to_numeric(master["elevation"], errors="coerce").fillna(0) - depth_vals

    if treo_col and treo_col in master.columns and master[treo_col].notna().any():
        size_col = "size_viz"
        master[size_col] = pd.to_numeric(master[treo_col], errors="coerce").fillna(0).clip(lower=0)
        if master[size_col].max() > 1000:
            master[size_col] = np.log1p(master[size_col])
    else:
        size_col = "score_100" 

    hover_col = "companyholeid" if "companyholeid" in master.columns else None

    fig_3d = px.scatter_3d(
        master, x="lon", y="lat", z="depth_z",
        color="score_100", size=size_col,
        hover_name=hover_col,
        labels={"depth_z": "Elevation (m)", "score_100": "AI Score"},
        color_continuous_scale="Viridis",
        template="plotly_dark",
    )
    fig_3d.update_layout(
        scene=dict(
            aspectmode="data",
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)")
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=700
    )
    st.plotly_chart(fig_3d, use_container_width=True)


def _generate_pdf(result: dict, master: pd.DataFrame) -> bytes:
    """Build an executive PDF report and return its bytes."""
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Arial", "B", 16)
    pdf.cell(190, 10, "Geo-AI-India Exploration Report", 0, 1, "C")
    pdf.set_font("Arial", "", 12)
    pdf.cell(190, 10,
             f"Deposit: {result['deposit'].upper()} | "
             f"Date: {result.get('trained_date', 'N/A')}", 0, 1, "C")
    pdf.ln(10)

    pdf.set_font("Arial", "B", 14)
    pdf.cell(100, 10, "Executive Summary", 0, 1)
    pdf.set_font("Arial", "", 11)
    pdf.multi_cell(
        180, 7,
        f"This deposit was analyzed using the Geo-AI-India ensemble pipeline. "
        f"The model achieved a CV R2 of {result['cv_r2']:.4f} with "
        f"{result.get('n_holes_total', result['n_labelled'])} holes. "
        f"20 high-priority targets have been identified for follow-up drilling.",
    )

    pdf.ln(5)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, "Top 5 Targets", 0, 1)
    pdf.set_font("Arial", "", 10)
    for _, t in master.nlargest(5, "score_100").iterrows():
        hole_id = t.get("companyholeid", "N/A")
        pdf.cell(
            180, 6,
            f"Hole: {hole_id} | Lon: {t['lon']:.4f} | "
            f"Lat: {t['lat']:.4f} | Score: {t['score_100']:.1f}",
            0, 1,
        )
    return bytes(pdf.output())


def _render_downloads(result: dict, master: pd.DataFrame, pipe) -> None:
    """Render the three download buttons (CSV, GeoJSON, PDF)."""
    deposit = result["deposit"]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download results (CSV)",
            master.to_csv(index=False).encode(),
            f"geoai_results_{deposit}.csv", "text/csv",
            use_container_width=True,
        )
    with c2:
        geojson_data = pipe.to_geojson(master)
        st.download_button(
            "Download for QGIS (GeoJSON)",
            geojson_data.encode(),
            f"geoai_qgis_{deposit}.geojson", "application/json",
            use_container_width=True,
        )
    with c3:
        pdf_bytes = _generate_pdf(result, master)
        st.download_button(
            "Download CEO Report (PDF)",
            pdf_bytes,
            f"geoai_report_{deposit}.pdf", "application/pdf",
            use_container_width=True,
        )


def _render_model_insights(result: dict) -> None:
    """Render interactive model comparison and SHAP charts using Plotly."""
    has_scores = bool(result.get("model_scores"))
    has_shap   = bool(result.get("shap_values"))
    if not (has_scores or has_shap):
        return

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🧠 AI Model Intelligence")
    
    ci1, ci2 = st.columns(2)

    with ci1:
        if has_scores:
            st.markdown("#### Performance Benchmark")
            ms = dict(result["model_scores"])
            ms["Ensemble"] = result["cv_r2"]
            
            # Create Plotly bar chart
            df_ms = pd.DataFrame([{"Model": k.upper(), "R2 Score": v} for k, v in ms.items()])
            df_ms = df_ms.sort_values("R2 Score", ascending=True)
            
            fig = px.bar(
                df_ms, y="Model", x="R2 Score", orientation='h',
                color="R2 Score", color_continuous_scale="Viridis",
                template="plotly_dark", height=300
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)

    with ci2:
        if has_shap:
            st.markdown("#### Feature Influence (SHAP)")
            shap_data = result["shap_values"]
            df_shap = pd.DataFrame([{"Feature": k, "Impact": v} for k, v in shap_data.items()])
            df_shap = df_shap.sort_values("Impact", ascending=True).tail(10)
            
            fig_shap = px.bar(
                df_shap, y="Feature", x="Impact", orientation='h',
                template="plotly_dark", height=300,
                color_discrete_sequence=['#D4AF37']
            )
            fig_shap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_shap, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🛰️ Geo-AI India")
    st.markdown("<div style='font-size:0.8rem; color:#888; margin-top:-10px'>Strategic Exploration Engine</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("##### 🧬 Collective Intelligence")
    # Show trained deposits from registry
    try:
        reg_path = Path(OUTPUT_DIR) / "deposit_registry.json"
        if reg_path.exists():
            reg = json.loads(reg_path.read_text())
            deposits = reg.get("deposits", {})
            if deposits:
                for dep_name, info in deposits.items():
                    v = info["versions"][-1]
                    st.markdown(
                        f'<div style="background:rgba(212,175,55,0.1); border:1px solid rgba(212,175,55,0.2); '
                        f'border-radius:6px; padding:10px; margin-bottom:8px">'
                        f'<div style="font-size:0.7rem; color:#888; text-transform:uppercase">Deposit</div>'
                        f'<div style="font-size:0.9rem; color:var(--primary); font-weight:600">{dep_name.replace("_"," ").title()}</div>'
                        f'<div style="font-size:0.75rem; margin-top:4px">R² Score: <b>{v["cv_r2"]:.3f}</b></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
    except Exception:
        st.caption("Intelligence registry unavailable")

    st.markdown("---")
    st.markdown("##### 📦 Supported Archives")
    st.markdown(
        "<div style='font-size:0.75rem; color:#777'>"
        "● Drillhole (CSV, XLS, XLSX)<br>"
        "● Geophysics (TIF, ERS, GRD)<br>"
        "● Geology (SHP, GeoJSON)<br>"
        "● Satellite (JP2, ECW)<br>"
        "● Archives (ZIP, TAR, GZ)"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    st.caption(f"Engine v{VERSION}")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("<h1>Geo-AI India — Strategic Intelligence</h1>", unsafe_allow_html=True)
st.markdown(
    '<div style="color:#AAA; font-size:1.1rem; margin-bottom:2rem; max-width:800px">'
    "Harnessing the world's leading mineral AI to de-risk exploration. "
    "Upload raw geological, geophysical, or satellite data to generate "
    "precision-targeted prospectivity maps."
    "</div>", 
    unsafe_allow_html=True
)

# ═══════════════════════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════════════════════
with st.container():
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 🛰️ Data Ingestion Portal")
    col_upload, col_name = st.columns([3, 1])
    with col_upload:
        uploaded = st.file_uploader(
            "Drop your assets here (CSV, TIF, SHP, ZIP, etc.)",
            accept_multiple_files=True, type=None, label_visibility="collapsed",
        )
    with col_name:
        dep_name = st.text_input("Deposit Name", placeholder="e.g. mountain_pass")
    st.markdown('</div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE EXECUTION
# ═══════════════════════════════════════════════════════════════
if uploaded:
    from geoai.categoriser import categorise_file
    from collections import Counter

    cats = []
    for f in uploaded:
        suf = Path(f.name).suffix
        tmp = tempfile.NamedTemporaryFile(suffix=suf, delete=False)
        tmp.write(f.read())
        tmp.close()
        r = categorise_file(tmp.name)
        r["tmp_path"] = tmp.name
        r["original_name"] = f.name
        cats.append(r)

    # Show categorisation badges
    st.markdown(f"**{len(uploaded)} file(s) uploaded**")
    cols = st.columns(min(len(cats), 4))
    for i, r in enumerate(cats):
        bg, fg = LAYER_COLOURS.get(r["layer"], ("#333", "#eee"))
        with cols[i % 4]:
            st.markdown(
                f'<div class="pill">{r["original_name"][:22]}<br>'
                f'<span class="badge" style="background:{bg};color:{fg}">'
                f'{r["layer"]}</span></div>',
                unsafe_allow_html=True,
            )

    lc = Counter(r["layer"] for r in cats)
    html_badges = "".join(
        f'<span class="badge" style="background:{LAYER_COLOURS.get(layer, ("#333","#eee"))[0]};'
        f'color:{LAYER_COLOURS.get(layer, ("#333","#eee"))[1]}">{layer}: {count}</span>'
        for layer, count in lc.items()
    )
    st.markdown(html_badges, unsafe_allow_html=True)

    # ── Run pipeline ──────────────────────────────────────────
    if st.button("Run prospectivity analysis"):
        log_lines: list[str] = []
        log_ph = st.empty()

        def upd_log(msg: str) -> None:
            log_lines.append(f"› {msg}")
            log_ph.markdown(
                '<div class="logbox">' + "<br>".join(log_lines[-18:]) + "</div>",
                unsafe_allow_html=True,
            )

        tmp_dir = Path(tempfile.mkdtemp())
        fps = []
        for r in cats:
            dst = tmp_dir / r["original_name"]
            shutil.copy(r["tmp_path"], str(dst))
            fps.append(dst)

        try:
            from geoai.pipeline import GeoAIPipeline

            pipe = GeoAIPipeline(output_dir=OUTPUT_DIR)
            result = pipe.run(
                files=fps,
                deposit_name=dep_name.strip() or None,
                inference_only=True,
                progress_cb=upd_log,
            )
        except Exception as e:
            import traceback
            st.error(f"Pipeline error: {e}")
            st.code(traceback.format_exc())
            result = {"status": "error"}
        finally:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)

        # ── Render results ─────────────────────────────────────
        if result.get("status") == "success":
            st.success("Analysis complete!")
            master   = result["master_df"]
            treo_col = result["treo_col"]

            try:
                _render_metrics(result)
            except Exception as e:
                st.warning(f"Metrics display error: {e}")

            try:
                _render_data_health(master, treo_col)
            except Exception as e:
                st.warning(f"Data health error: {e}")

            try:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🗺️ Prospectivity Heatmap")
                _render_map(master, treo_col)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Map rendering error: {e}")

            try:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🎯 Priority Drill Targets")
                show_cols = [
                    c for c in ["companyholeid", "lat", "lon", "score_100",
                                treo_col, "depth_score"]
                    if c and c in master.columns
                ]
                st.dataframe(
                    master.nlargest(20, "score_100")[show_cols],
                    use_container_width=True
                )
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Targets table error: {e}")

            try:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### 🌐 3D Sub-surface Explorer")
                _render_3d_visualiser(master, treo_col)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"3D visualizer error: {e}")

            st.markdown("---")

            try:
                _render_downloads(result, master, pipe)
            except Exception as e:
                st.warning(f"Downloads error: {e}")
                import traceback
                st.code(traceback.format_exc())

            try:
                _render_model_insights(result)
            except Exception as e:
                st.warning(f"Model insights error: {e}")

        elif result.get("status") == "insufficient_data":
            st.warning(
                f"Only {result['n_labelled']} labelled samples found. "
                "Upload assay data with TREO grades to enable training."
            )

else:
    # ── Empty state (Landing Page) ────────────────────────────
    st.markdown("""
    <div style="background: rgba(212, 175, 55, 0.05); border: 1px dashed rgba(212, 175, 55, 0.3); 
                border-radius: 16px; padding: 4rem 2rem; text-align: center; margin-top: 2rem">
      <div style="font-size: 3.5rem; margin-bottom: 1.5rem">💎</div>
      <h2 style="color: #FFF; margin-bottom: 1rem">Ready for Discovery</h2>
      <p style="color: #888; font-size: 1.1rem; margin-bottom: 2rem; max-width: 600px; margin-left: auto; margin-right: auto">
        Our multi-deposit intelligence engine is primed. Upload your exploration files 
        above to generate investment-grade targeting reports in minutes.
      </p>
      <div style="display: flex; justify-content: center; gap: 2rem; color: var(--primary); font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; letter-spacing: 1px">
        <span>● DRILLHOLE</span>
        <span>● GEOPHYSICS</span>
        <span>● RADIOMETRICS</span>
        <span>● SATELLITE</span>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### Feature Coverage")
    st.dataframe(
        pd.DataFrame({
            "Source Layer":    ["Geochemical", "Geophysical", "Remotely Sensed", "Geological", "Topographic"],
            "Capabilities":    ["Multi-element assaying, LREE/HREE ratios", "Gravity, Magnetics, Radiometrics", "ASTER, Sentinel, Landsat Indices", "Structural mapping, Lithology", "DEM, Slope, Aspect analysis"],
            "AI Readiness":    ["High", "High", "Medium", "Medium", "Low"]
        }),
        use_container_width=True, hide_index=True,
    )

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
}


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════
def _render_metrics(result: dict) -> None:
    """Render the top-level KPI metrics row."""
    st.markdown(
        f'<div style="display:flex;gap:1rem;margin:1rem 0">'
        f'<div class="mbox"><div class="mval">{result["cv_r2"]:.3f}</div>'
        f'<div class="mlbl">CV R²</div></div>'
        f'<div class="mbox"><div class="mval">{result["roc_auc"]:.3f}</div>'
        f'<div class="mlbl">ROC AUC</div></div>'
        f'<div class="mbox"><div class="mval">{result["n_labelled"]}</div>'
        f'<div class="mlbl">Labelled holes</div></div>'
        f'<div class="mbox"><div class="mval">{result["top_score"]:.0f}</div>'
        f'<div class="mlbl">Top score</div></div>'
        f'<div class="mbox"><div class="mval">{result["n_features"]}</div>'
        f'<div class="mlbl">Features</div></div></div>',
        unsafe_allow_html=True,
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
    """Render the 3D sub-surface explorer."""
    with st.expander("🌐 3D Sub-surface Visualizer", expanded=False):
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

        # Ensure size_col is valid for Plotly (must be >= 0 and non-NaN)
        if treo_col and treo_col in master.columns and master[treo_col].notna().any():
            size_col = "size_viz"
            master[size_col] = pd.to_numeric(master[treo_col], errors="coerce").fillna(0).clip(lower=0)
            # Normalize for visualization if too large
            if master[size_col].max() > 1000:
                master[size_col] = np.log1p(master[size_col])
        else:
            size_col = "score_100" # Fallback to AI score for sizing

        hover_col = "companyholeid" if "companyholeid" in master.columns else None

        fig_3d = px.scatter_3d(
            master, x="lon", y="lat", z="depth_z",
            color="score_100", size=size_col,
            hover_name=hover_col,
            labels={"depth_z": "Elevation (m)"},
            color_continuous_scale="Turbo",
            title="Sub-surface Prospectivity Heatmap",
        )
        fig_3d.update_layout(scene=dict(aspectmode="data"), height=600)
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
    """Render model comparison and SHAP / feature importance charts."""
    has_scores = bool(result.get("model_scores"))
    has_shap   = bool(result.get("shap_values"))
    if not (has_scores or has_shap):
        return

    st.markdown("---")
    ci1, ci2 = st.columns(2)

    with ci1:
        if has_scores:
            confidence = result.get("confidence", 0) * 100
            st.markdown(f"#### Model confidence: **{confidence:.1f}%**")
            st.markdown("#### Model comparison")
            ms = dict(result["model_scores"])
            ms["Ensemble"] = result["cv_r2"]
            for name, r2 in sorted(ms.items(), key=lambda x: -x[1]):
                bar = "█" * max(0, int(r2 * 30))
                st.markdown(f"`{name.upper():<10}` **{r2:+.4f}** {bar}")

    with ci2:
        if has_shap:
            st.markdown("#### AI Reasoning (SHAP Impact)")
            shap_data = result["shap_values"]
            max_val = max(shap_data.values()) if shap_data else 1
            for feat, val in sorted(shap_data.items(), key=lambda x: -x[1]):
                bar = "█" * max(1, int(val / max_val * 40))
                st.markdown(f"`{feat[:18]:<18}` {bar}")


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⛏️ Geo-AI-India")
    st.markdown("**Mineral Exploration AI**")
    st.markdown("---")
    st.markdown("**Accepts any format**")
    for line in [
        "Drillhole: CSV, XLS, XLSX", "Geophysics: TIF, ERS, GRD",
        "Geology: SHP, GeoJSON", "Satellite: JP2, ECW",
        "Topography: ASC, XYZ", "Archives: ZIP, TAR",
    ]:
        st.markdown(f"- {line}")
    st.markdown("---")

    # Show trained deposits from registry
    try:
        reg_path = Path(OUTPUT_DIR) / "deposit_registry.json"
        if reg_path.exists():
            reg = json.loads(reg_path.read_text())
            deposits = reg.get("deposits", {})
            if deposits:
                st.markdown("**Trained deposits:**")
                for dep_name, info in deposits.items():
                    v = info["versions"][-1]
                    bg, fg = LAYER_COLOURS["geophysics"]
                    st.markdown(
                        f'<span class="badge" style="background:{bg};color:{fg}">'
                        f"{dep_name}</span> R²={v['cv_r2']}",
                        unsafe_allow_html=True,
                    )
    except Exception:
        pass

    st.caption(f"v{VERSION}")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("# Geo-AI-India — Mineral Exploration AI")
st.markdown(
    "The future of intelligent mineral targeting. Upload raw exploration data "
    "from any deposit to receive a geologically-validated prospectivity map "
    "and ranked drill targets."
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════════════════════
st.markdown("### Upload deposit data")
col_upload, col_name = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Any files — CSV, TIF, SHP, ZIP, XLS, JP2, or anything else",
        accept_multiple_files=True, type=None, label_visibility="collapsed",
    )
with col_name:
    dep_name = st.text_input("Deposit name (optional)",
                             placeholder="e.g. mount_weld")

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
                st.markdown("#### Prospectivity map")
                _render_map(master, treo_col)
            except Exception as e:
                st.warning(f"Map rendering error: {e}")

            try:
                st.markdown("#### Top 20 drill targets")
                show_cols = [
                    c for c in ["companyholeid", "lat", "lon", "score_100",
                                treo_col, "depth_score"]
                    if c and c in master.columns
                ]
                st.dataframe(
                    master.nlargest(20, "score_100")[show_cols],
                )
            except Exception as e:
                st.warning(f"Targets table error: {e}")

            try:
                _render_3d_visualiser(master, treo_col)
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
    # ── Empty state ────────────────────────────────────────────
    st.markdown("""
<div style="border:1px dashed #3D2E14;border-radius:8px;padding:3rem;
            text-align:center;color:#6B5535;margin:2rem 0">
  <div style="font-size:2rem;margin-bottom:1rem">⛏️</div>
  <div style="font-family:'IBM Plex Mono',monospace;font-size:.9rem;margin-bottom:.5rem">
    Upload exploration data above to begin
  </div>
  <div style="font-size:.8rem">CSV · TIF · SHP · JP2 · XLS · ZIP · TAR · any format</div>
</div>""", unsafe_allow_html=True)

    st.markdown("#### What to upload")
    st.dataframe(
        pd.DataFrame({
            "File type":       ["Collar CSV", "Assay CSV", "Geophysics TIF",
                                "Geology SHP", "Satellite JP2"],
            "Auto-detected as": ["drillhole", "drillhole", "geophysics",
                                 "geology", "satellite"],
            "Required?":       ["Yes", "Recommended", "Optional",
                                "Optional", "Optional"],
        }),
        use_container_width=True, hide_index=True,
    )

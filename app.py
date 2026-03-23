"""
=================================================================
  GeoAI — REE Prospectivity Engine
  Streamlit Web Application

  Deploy free:  streamlit run app.py
  Cloud deploy: https://streamlit.io/cloud (free tier)

  Loads the saved ree_model_bundle.joblib and serves predictions
  through a browser interface. No retraining needed.
=================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from pathlib import Path
from streamlit_folium import st_folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
import datetime

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "GeoAI — REE Prospectivity",
    page_icon   = "🪨",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
  }
  .main { background: #0d0a06; }
  .block-container { padding: 1.5rem 2rem; max-width: 1400px; }

  h1 { font-family: 'IBM Plex Mono', monospace !important;
       color: #C9A84C !important; letter-spacing: -0.5px; }
  h2, h3 { color: #E8D5A0 !important; }

  .metric-card {
    background: #1a1208;
    border: 1px solid #3D2E14;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    margin: 0.3rem 0;
  }
  .metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    color: #C9A84C;
    line-height: 1;
  }
  .metric-label {
    font-size: 0.75rem;
    color: #6B5535;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
  }
  .deposit-tag {
    display: inline-block;
    background: #3D2E14;
    color: #C9A84C;
    border-radius: 4px;
    padding: 2px 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    margin: 2px;
  }
  .stDataFrame { border: 1px solid #3D2E14 !important; }
  .stButton > button {
    background: #C9A84C;
    color: #0d0a06;
    border: none;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 500;
    letter-spacing: 0.5px;
    border-radius: 4px;
    padding: 0.5rem 1.5rem;
  }
  .stButton > button:hover { background: #E8D5A0; }
  .upload-zone {
    border: 1px dashed #3D2E14;
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    color: #6B5535;
  }
  div[data-testid="stSidebar"] {
    background: #0d0a06;
    border-right: 1px solid #1a1208;
  }
</style>
""", unsafe_allow_html=True)

# ── DEFAULT MODEL PATHS ───────────────────────────────────────
DEFAULT_BUNDLE_PATHS = [
    r"D:\GeoAI-INDIA\ree_output\ree_model_bundle.joblib",
    Path(__file__).parent / "ree_model_bundle.joblib",
    Path(__file__).parent / "model" / "ree_model_bundle.joblib",
]

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model bundle...")
def load_model(path):
    return joblib.load(str(path))

def find_bundle():
    for p in DEFAULT_BUNDLE_PATHS:
        if Path(str(p)).exists():
            return Path(str(p))
    return None

# ── PREDICT FUNCTION ──────────────────────────────────────────
def predict_scores(bundle, df_input):
    """
    Given a dataframe with drillhole data, return prospectivity scores.
    Missing features are filled with 0 (safe default).
    """
    feat_cols = bundle["feat_cols"]
    scaler    = bundle["scaler"]
    pca       = bundle["pca"]
    models    = bundle["models"]
    meta      = bundle["meta"]
    p95       = bundle["p95_treo"]

    # Build feature matrix aligned to training schema
    X = pd.DataFrame(0.0, index=df_input.index, columns=feat_cols)
    for col in feat_cols:
        col_lower = col.lower()
        # Try exact match, then case-insensitive
        if col in df_input.columns:
            vals = pd.to_numeric(df_input[col], errors="coerce").fillna(0)
        elif col_lower in df_input.columns:
            vals = pd.to_numeric(df_input[col_lower], errors="coerce").fillna(0)
        else:
            vals = pd.Series(0.0, index=df_input.index)
        # log1p transform geochemical columns (must match training transform)
        if "_ppm" in col or "ratio" in col or col in ["lree_max","hree_max","treo_max"]:
            vals = np.log1p(vals.clip(lower=0))
        X[col] = vals.values

    X_np  = X.values.astype(float)
    X_s   = scaler.transform(X_np)
    X_p   = pca.transform(X_s)

    col_order = sorted(models.keys())
    preds = []
    for k in col_order:
        m = models[k]
        p = m.predict(X_np if hasattr(m, "steps") else X_p).clip(0, 1)
        preds.append(p)

    meta_X = np.column_stack(preds)
    scores = meta.predict(meta_X).clip(0, 1)
    return (scores * 100).round(1)

# ── MAP BUILDER ───────────────────────────────────────────────
def build_map(df_scored):
    if "lat" not in df_scored.columns or "lon" not in df_scored.columns:
        return None
    valid = df_scored.dropna(subset=["lat","lon"])
    if len(valid) == 0:
        return None
    clat = float(valid["lat"].median())
    clon = float(valid["lon"].median())
    m = folium.Map(location=[clat, clon], zoom_start=14,
                   tiles="CartoDB dark_matter")
    from folium.plugins import HeatMap, MarkerCluster
    heat_data = [[float(r.lat), float(r.lon), float(r.score_100)/100]
                 for r in valid.itertuples() if np.isfinite(r.lat)]
    HeatMap(heat_data, radius=16, blur=12,
            gradient={0:"#1a0a00", 0.4:"#5C2010", 0.65:"#A0400A",
                      0.8:"#D4601A", 0.92:"#C9A84C", 1:"#FFF5D0"},
            min_opacity=0.4).add_to(m)
    mc = MarkerCluster(name="Top targets").add_to(m)
    for i, row in valid.nlargest(30, "score_100").iterrows():
        sc = float(row["score_100"])
        color = "#C9A84C" if sc >= 75 else "#D4601A" if sc >= 60 else "#8B4513"
        treo_str = f"{row['treo_max']:,.0f} ppm" if "treo_max" in row and pd.notna(row.get("treo_max")) else "predicted"
        folium.CircleMarker(
            location=[float(row["lat"]), float(row["lon"])],
            radius=9 if sc >= 75 else 6,
            color=color, fill=True, fill_opacity=0.85,
            popup=folium.Popup(
                f"<b>{row.get('companyholeid','Hole')}</b><br>"
                f"Score: <b>{sc:.0f}/100</b><br>TREO: {treo_str}",
                max_width=200),
            tooltip=f"{row.get('companyholeid','')}  {sc:.0f}/100"
        ).add_to(mc)
    folium.LayerControl().add_to(m)
    return m

# ══════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════
def main():

    # ── SIDEBAR ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🪨 GeoAI")
        st.markdown("**REE Prospectivity Engine**")
        st.markdown("---")

        # Model loading
        st.markdown("### Model")
        bundle_path = find_bundle()
        if bundle_path:
            st.success(f"Bundle found")
            bundle = load_model(bundle_path)
            info   = bundle["meta_info"]
            st.markdown(f"**Version:** `{info.get('version','?')}`")
            st.markdown(f"**Trained:** {info.get('trained_date','?')}")
            st.markdown("**Deposits:**")
            for d in info.get("deposits", []):
                st.markdown(f'<span class="deposit-tag">{d}</span>', unsafe_allow_html=True)
            st.markdown(f"**Training holes:** {info.get('n_holes_labelled','?')}")
        else:
            st.error("No model bundle found")
            st.markdown("Run `step3_map.py` first to generate `ree_model_bundle.joblib`")
            bundle = None
            info   = {}

        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
Global REE mineral prospectivity engine.
Currently trained on Mount Weld, WA.

**CV R²** = 0.853  
**ROC AUC** = 0.952

*Data flywheel: each new deposit added improves the global model.*
        """)
        st.markdown("---")
        st.markdown("**[GitHub](https://github.com)** · MIT License")

    # ── HEADER ───────────────────────────────────────────────
    st.markdown("# GeoAI — REE Prospectivity Engine")
    st.markdown(
        "AI-powered rare earth element mineral targeting. "
        "Upload drillhole data → get a ranked prospectivity map."
    )

    if bundle is None:
        st.stop()

    # ── METRICS ROW ──────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{info.get('cv_r2', 0.853):.3f}</div>
        <div class="metric-label">CV R² Score</div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{info.get('roc_auc', 0.952):.3f}</div>
        <div class="metric-label">ROC AUC</div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{info.get('n_holes_labelled', 119)}</div>
        <div class="metric-label">Training holes</div></div>""", unsafe_allow_html=True)
    with col4:
        n_deposits = len(info.get("deposits", ["mount_weld"]))
        st.markdown(f"""<div class="metric-card">
        <div class="metric-value">{n_deposits}</div>
        <div class="metric-label">Deposits in model</div></div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── TWO MODES ────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "  Predict — upload your data  ",
        "  Mount Weld results  ",
        "  Model details  "
    ])

    # ── TAB 1: PREDICT ───────────────────────────────────────
    with tab1:
        st.markdown("### Upload drillhole data")
        st.markdown(
            "Upload a CSV with drillhole coordinates and any assay data you have. "
            "The model will predict prospectivity scores for all holes."
        )

        col_a, col_b = st.columns(2)
        with col_a:
            collar_file = st.file_uploader(
                "Collar file (required)",
                type=["csv"],
                help="Must contain: companyholeid, latitude, longitude"
            )
        with col_b:
            assay_file = st.file_uploader(
                "Assay file (optional — improves predictions)",
                type=["csv"],
                help="Must contain: companyholeid + REE element columns"
            )

        if collar_file:
            collar_df = pd.read_csv(collar_file)
            collar_df.columns = [c.strip().lower() for c in collar_df.columns]
            st.success(f"Collar loaded: {len(collar_df)} holes")

            # Merge assay if provided
            if assay_file:
                assay_df = pd.read_csv(assay_file)
                assay_df.columns = [c.strip().lower() for c in assay_df.columns]
                # Pivot if unpivoted
                if "attributecolumn" in assay_df.columns:
                    try:
                        assay_pivot = assay_df.pivot_table(
                            index="companyholeid",
                            columns="attributecolumn",
                            values="attributevalue",
                            aggfunc="max"
                        ).reset_index()
                        assay_pivot.columns = [str(c).lower()+"_ppm"
                                               if c != "companyholeid" else c
                                               for c in assay_pivot.columns]
                        collar_df = collar_df.merge(assay_pivot,
                                                    on="companyholeid", how="left")
                    except Exception:
                        collar_df = collar_df.merge(assay_df,
                                                    on="companyholeid", how="left")
                else:
                    collar_df = collar_df.merge(assay_df,
                                                on="companyholeid", how="left")
                st.success(f"Assay merged: {len(collar_df)} holes")

            if st.button("Run prospectivity prediction"):
                with st.spinner("Computing scores..."):
                    scores = predict_scores(bundle, collar_df)
                    collar_df["score_100"] = scores
                    result = collar_df.copy()

                st.success(f"Done! {len(result)} holes scored.")

                # Display map
                map_obj = build_map(result)
                if map_obj:
                    st.markdown("#### Prospectivity map")
                    st_folium(map_obj, width=None, height=500)

                # Top targets table
                st.markdown("#### Top 20 drill targets")
                show_cols = [c for c in ["companyholeid","lat","lon","score_100"]
                             if c in result.columns]
                top20 = result.nlargest(20, "score_100")[show_cols]
                st.dataframe(top20, use_container_width=True)

                # Download
                csv_bytes = result.to_csv(index=False).encode()
                st.download_button(
                    "Download full results (CSV)",
                    data     = csv_bytes,
                    file_name= f"ree_prospectivity_{datetime.date.today()}.csv",
                    mime     = "text/csv"
                )
        else:
            st.markdown("""
<div class="upload-zone">
Upload a collar CSV to get started.<br><br>
<small>Required columns: <code>companyholeid</code>, <code>latitude</code>, <code>longitude</code></small>
</div>""", unsafe_allow_html=True)

            st.markdown("#### Example CSV format")
            example = pd.DataFrame({
                "companyholeid": ["RC001","RC002","RC003"],
                "latitude":      [-28.861, -28.862, -28.863],
                "longitude":     [122.545, 122.546, 122.547],
            })
            st.dataframe(example, use_container_width=True)

    # ── TAB 2: MOUNT WELD ────────────────────────────────────
    with tab2:
        st.markdown("### Mount Weld — training deposit results")
        st.markdown(
            "These results are from the Mount Weld carbonatite deposit, "
            "Western Australia — the world's highest-grade REE mine."
        )

        # Try load pre-computed results
        result_paths = [
            r"D:\GeoAI-INDIA\ree_output\scored_data.csv",
            Path(__file__).parent / "outputs" / "scored_data.csv",
        ]
        scored_df = None
        for rp in result_paths:
            if Path(str(rp)).exists():
                scored_df = pd.read_csv(str(rp), low_memory=False)
                break

        if scored_df is not None:
            col_l, col_r = st.columns([3, 2])
            with col_l:
                map_mw = build_map(scored_df)
                if map_mw:
                    st_folium(map_mw, width=None, height=480)
            with col_r:
                treo_col = next((c for c in scored_df.columns
                                 if "treo" in c and "max" in c), None)
                st.markdown("#### Top 15 targets")
                show = [c for c in ["companyholeid","score_100", treo_col, "depth_score"]
                        if c and c in scored_df.columns]
                top15 = scored_df.nlargest(15, "score_100")[show]
                st.dataframe(top15, use_container_width=True)

                st.markdown("#### Score distribution")
                fig, ax = plt.subplots(figsize=(5,3), facecolor="#0d0a06")
                ax.set_facecolor("#1a1208")
                ax.hist(scored_df["score_100"], bins=30,
                        color="#C9A84C", alpha=0.85, edgecolor="#0d0a06")
                ax.axvline(70, color="#E24B4A", lw=1.5, ls="--", label="70 threshold")
                ax.set_xlabel("Prospectivity score", color="#6B5535", fontsize=9)
                ax.set_ylabel("Holes", color="#6B5535", fontsize=9)
                ax.tick_params(colors="#6B5535", labelsize=8)
                for sp in ax.spines.values(): sp.set_color("#3D2E14")
                ax.legend(facecolor="#1a1208", labelcolor="#C9A84C", fontsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
        else:
            st.info(
                "Run step3_map.py first to generate results, then reload this page."
            )

    # ── TAB 3: MODEL DETAILS ─────────────────────────────────
    with tab3:
        st.markdown("### Model details")
        col_i, col_ii = st.columns(2)
        with col_i:
            st.markdown("#### Architecture")
            st.code("""
RF    CV R² = 0.724  (500 trees, OOB validation)
SVM   CV R² = 0.863  (RBF kernel, C=10)
XGB   CV R² = 0.817  (400 estimators, depth=6)
────────────────────────────────
ENSEMBLE  0.853  (Ridge stacking)

Meta weights:
  SVM  47.7%  ← geochemistry signal
  XGB  35.1%  ← depth + alteration
  RF   23.2%  ← structural geophys
            """, language="text")

        with col_ii:
            st.markdown("#### Geological validation")
            st.markdown("""
| Check | Result |
|-------|--------|
| TREO grade range (43.7% max) | ✅ Matches CLD crown |
| Ore depth 0–54m | ✅ Matches laterite profile |
| LREE/HREE ratio = 30.3 | ✅ LREE-dominant deposit |
| Top target at pipe centre | ✅ Matches known geology |
| Laterite + goethite pattern | ✅ Matches ore mechanism |
            """)
            st.markdown("*Validated against: Lynas 2024, USGS, Cook et al. 2023*")

        st.markdown("#### Feature importance")
        feat_names = bundle.get("feat_cols", [])
        try:
            rf_model = bundle["models"].get("rf")
            if hasattr(rf_model, "feature_importances_"):
                fi = rf_model.feature_importances_
                top_n = 15
                idx = np.argsort(fi)[-top_n:][::-1]
                top_feats = [(feat_names[i] if i < len(feat_names) else f"feat_{i}",
                              float(fi[i])) for i in idx]
                fig2, ax2 = plt.subplots(figsize=(8,4), facecolor="#0d0a06")
                ax2.set_facecolor("#1a1208")
                names = [f[0][:28] for f in top_feats][::-1]
                vals  = [f[1] for f in top_feats][::-1]
                ax2.barh(names, vals, color="#C9A84C", alpha=0.85)
                ax2.tick_params(colors="#6B5535", labelsize=8)
                for sp in ax2.spines.values(): sp.set_color("#3D2E14")
                ax2.set_xlabel("Importance", color="#6B5535", fontsize=9)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
        except Exception:
            st.markdown("Feature importance chart requires a fitted RF model in the bundle.")

        st.markdown("#### Deposit coverage")
        st.markdown("*Currently trained on:*")
        for d in info.get("deposits", ["mount_weld"]):
            st.markdown(f'<span class="deposit-tag">{d}</span>', unsafe_allow_html=True)
        st.markdown("""
*Roadmap:*
- Browns Range (HREE-rich, WA)  
- Ngualla (Tanzania carbonatite)  
- Mountain Pass (California)  
- Bayan Obo (China — world's largest)
        """)


if __name__ == "__main__":
    main()

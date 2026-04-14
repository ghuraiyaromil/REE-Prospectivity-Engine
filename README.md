# Geo-AI-India — Mineral Exploration AI

> **Professional-grade AI targeting for modern mineral exploration.**
> Transforming raw geological datasets into investor-ready prospectivity maps and ranked drill targets.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Geo--AI--India-Dashboard-FF4B4B?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E)](LICENSE)

---

## ⛏️ The Platform

**Geo-AI-India** is a portable, modular AI targeting engine designed for rapid mineral exploration. It bridges the gap between raw geoscientific field data and executive-level decision making. By combining multi-physics ensembles with explainable AI (XAI), we provide high-confidence targeting for REE, gold, and base metal systems.

---

## 🚀 Key Features

### 1. Unified Intake & Auto-Pipeline
Upload **any format** — Drillhole CSVs, Geophysics TIFs, Geology SHPs, or Satellite JP2s. Our automated pipeline handles coordinate re-projection, feature engineering (log1p, PCA), and IDW pseudo-labelling out of the box.

### 2. High-Performance Ensemble ML (Leakage-Free)
Our "Global Model" utilizes **Spatial GroupKFold** cross-validation within nested scikit-learn Pipelines (Imputer -> SelectKBest -> Scaler -> Model) to completely eliminate data leakage. The ensemble relies on **Random Forest and Gradient Boosting** stacked via a Ridge meta-learner.
- **Honest Spatial R²**: ~ 0.68 (Geologically validated generalization on unseen terrain)
- **Honest Spatial ROC AUC**: > 0.93
- **Confidence Scoring**: Automatic uncertainty quantification based on ensemble variance.

### 3. Explainable AI (SHAP Reasoning)
Move beyond "black box" machine learning. Geo-AI-India integrates **SHAP (SHapley Additive exPlanations)** to show exactly which geological markers (e.g., LREE/HREE ratios, Magnetic highs, Potassic alteration) are driving each target.

### 4. Professional GIS & Reporting
- **QGIS Integration**: Export results directly to GeoJSON for native GIS workflows.
- **3D Sub-surface Visualizer**: Map drillhole prospectivity in 3D space using Plotly.
- **Investor Reports**: Generate one-click, professional **PDF Executive Summaries** with top targets and model metrics.
- **Data Health Dashboard**: Real-time QA/QC to flag coordinate errors or assay outliers before training.

---

## 🛠️ Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/ghuraiyaromil/REE-Prospectivity-Engine.git
cd REE-Prospectivity-Engine

# Install dependencies
pip install -r requirements.txt

# Launch the platform
streamlit run app.py
```

### Usage
1. **Launch**: Run `streamlit run app.py`.
2. **Upload**: Drag and drop your deposit files (Zip, CSV, TIF, etc.).
3. **Analyze**: The system categorizes your data and builds the feature matrix automatically.
4. **Target**: Explore the interactive heatmap, verify targets in 3D, and download your **CEO Report**.

---

## 🏗️ Technical Stack

- **Core**: Python 3.11+
- **ML**: Scikit-Learn, XGBoost, SHAP
- **Spatial**: Rasterio, PyProj, Geopandas
- **UI**: Streamlit, Folium, Plotly
- **Reports**: FPDF2

---

## 📂 Background Auto-Training
For industrial workflows, use the **Watcher** script to monitor folders for new deposit data:

```bash
python watch_and_train.py
```

Drops a folder in `data/deposits/` and the system will automatically extract, inventory, process, and train a new model version.

---

## 📜 License
MIT — see [LICENSE](LICENSE)

---
*Built for the future of Indian mineral exploration by Geo-AI-India.*

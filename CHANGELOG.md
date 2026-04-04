# Changelog

## [v9.0] - 2026-03-22 (Current)

### Added
- Sentinel-2 L2A and Landsat 9 L2SP satellite band integration
- 6 spectral indices: NDVI, Iron Oxide, Ferric Iron, Clay Index, NDWI, Alteration
- Y4 Yilgarn ML geophysics grids as structural context features
- 10m prediction grid (2,436 points) replacing drillhole-only prediction
- Spatial coordinate features to break Landsat pixel ties
- Cubic interpolation smoothing for heatmap surface
- joblib model bundle persistence (0.6 MB, loads in 3 seconds)
- retrain.py for incremental updates when new deposit data arrives
- Streamlit web app (app.py)

### Fixed
- PCA collapsed to 1 component: fixed n_components=15
- log1p transform prevents p2o5 scale explosion (76000x larger than CeO2)
- XGBoost, GradientBoosting, MLPRegressor positional argument crashes
- Identical scores from same Landsat pixel: spatial coords as features
- Blocky prospectivity map: cubic interpolation smoothing added
- CNN model dropped (Landsat cloud mask over deposit, R2=-0.028)
- master prospectivity length mismatch (677 vs 119)

## [v8.0] - 2026-03-21
- XGBoost added as third base model
- Ridge stacking ensemble (meta-learner)
- 10m grid prediction expanding coverage 3x vs drillholes

## [v6.0] - 2026-03-20
- CV R2 corrected from -1.236 to 0.685 via PCA fix and log1p transform

## [v1.0] - 2026-03-20
- Initial Random Forest pipeline on Mount Weld drillhole data
- Folium interactive heatmap
- Matplotlib static prospectivity map

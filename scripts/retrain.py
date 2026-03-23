"""
=================================================================
  REE ENGINE — INCREMENTAL RETRAIN
  retrain.py

  USE CASES:
  1. New drillholes added to an existing deposit (e.g. Mount Weld
     infill campaign, or data you missed initially)
  2. A brand new deposit added to the global model (e.g. Browns
     Range, Ngualla, Mountain Pass)

  HOW IT WORKS:
  - Loads the existing trained model bundle (ree_model_bundle.joblib)
  - Appends new training rows to the cached X_train / y_train
  - Retrains all base models + meta-learner on the combined data
  - Saves a new versioned bundle (never overwrites the old one)
  - Reports exactly what changed in metrics

  USAGE EXAMPLES:
  # Add new holes to Mount Weld (same deposit, new data):
  python retrain.py --mode update_deposit ^
      --deposit mount_weld ^
      --collar D:\new_data\new_collar.csv ^
      --assay  D:\new_data\new_assay_pivoted.csv

  # Add a completely new deposit (Browns Range):
  python retrain.py --mode new_deposit ^
      --deposit browns_range ^
      --collar D:\browns_range\collar.csv ^
      --assay  D:\browns_range\assay_pivoted.csv ^
      --geophys D:\browns_range\geophysics\

=================================================================
"""
import argparse, sys, warnings, datetime, hashlib
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

# ── CONFIG ────────────────────────────────────────────────────
MODEL_DIR    = r"D:\GeoAI-INDIA\ree_output"
BASE_EXTRACT = r"D:\GeoAI-INDIA\training_data_extracted"
BASE_ORIG    = r"D:\GeoAI-INDIA\training_data"
# ──────────────────────────────────────────────────────────────

def find_latest_bundle(model_dir):
    """Find the most recent model bundle in the output folder."""
    p = Path(model_dir)
    bundles = sorted(p.glob("ree_model_bundle*.joblib"), key=lambda f: f.stat().st_mtime)
    if not bundles:
        raise FileNotFoundError(f"No model bundle found in {model_dir}\nRun step3_map.py first.")
    return bundles[-1]

def load_bundle(path):
    print(f"  Loading bundle: {path.name}")
    bundle = joblib.load(str(path))
    info   = bundle["meta_info"]
    print(f"  Version:   {info['version']}")
    print(f"  Trained:   {info['trained_date']}")
    print(f"  Deposits:  {info['deposits']}")
    print(f"  Holes:     {info['n_holes_labelled']}")
    print(f"  CV R²:     {info['cv_r2']:.4f}  ROC: {info.get('roc_auc', 'n/a')}")
    return bundle

def process_new_collar_assay(collar_path, assay_path, feat_cols):
    """
    Process new drillhole data into a feature matrix.
    Only extracts the subset of features the existing model knows about.
    Missing features are filled with 0 (safe default).
    """
    collar = pd.read_csv(collar_path, low_memory=False)
    assay  = pd.read_csv(assay_path,  low_memory=False)
    collar.columns = [c.strip().lower() for c in collar.columns]
    assay.columns  = [c.strip().lower() for c in assay.columns]

    # Build same feature set as training
    REE_COLS = [c for c in assay.columns if any(k in c for k in
                ['ceo2','la2o3','nd2o3','pr6','sm2','eu2','gd2o3','dy2o3',
                 'y2o3','er2o3','yb2o3','fe2o3','p2o5','al2o3','sio2',
                 'mn_ppm','tho2','u3o8','nb2o5'])]
    for c in REE_COLS:
        if c in assay.columns:
            assay[c] = pd.to_numeric(assay[c], errors='coerce')

    treo_c = [c for c in assay.columns if 'ceo2' in c or 'la2o3' in c or
              'nd2o3' in c or 'pr6' in c or 'sm2' in c]
    assay['treo'] = assay[treo_c].sum(axis=1, skipna=True) if treo_c else np.nan
    assay.loc[assay[treo_c].isna().all(axis=1), 'treo'] = np.nan if treo_c else None

    lree_c = [c for c in ['ceo2_ppm','la2o3_ppm','nd2o3_ppm','pr6o11_ppm','sm2o3_ppm']
              if c in assay.columns]
    hree_c = [c for c in ['gd2o3_ppm','dy2o3_ppm','y2o3_ppm','er2o3_ppm','yb2o3_ppm']
              if c in assay.columns]
    if lree_c: assay['lree'] = assay[lree_c].sum(axis=1, skipna=True)
    if hree_c: assay['hree'] = assay[hree_c].sum(axis=1, skipna=True)
    if lree_c and hree_c:
        assay['lree_hree_ratio'] = assay['lree'] / (assay['hree'] + 0.001)

    agg_d = {}
    for c in REE_COLS + ['treo','lree','hree']:
        if c in assay.columns: agg_d[c] = ['max','mean']
    for c in ['lree_hree_ratio']:
        if c in assay.columns: agg_d[c] = ['mean']
    agg_d['fromdepth'] = 'min'; agg_d['todepth'] = 'max'

    assay_agg = assay.groupby('companyholeid').agg(agg_d)
    assay_agg.columns = ['_'.join(c) for c in assay_agg.columns]
    assay_agg = assay_agg.reset_index()
    master    = collar.merge(assay_agg, on='companyholeid', how='left')

    treo_col = next((c for c in master.columns if 'treo' in c and 'max' in c), None)
    labeled  = master[treo_col].notna() if treo_col else pd.Series([False]*len(master))
    n_new    = labeled.sum()
    print(f"  New data: {len(master)} holes, {n_new} with TREO assay")

    if n_new == 0:
        print("  WARNING: no labelled (assayed) holes in new data — cannot retrain")
        return None, None

    # Build feature matrix aligned to existing feat_cols
    X_new = pd.DataFrame(0.0, index=master[labeled].index, columns=feat_cols)
    for col in feat_cols:
        if col in master.columns:
            vals = pd.to_numeric(master.loc[labeled, col], errors='coerce').fillna(0)
            if '_ppm' in col or 'ratio' in col or col in ['lree_max','hree_max','treo_max']:
                vals = np.log1p(vals.clip(lower=0))
            X_new[col] = vals.values

    treo_vals = pd.to_numeric(master.loc[labeled, treo_col], errors='coerce')
    p95_new   = treo_vals.quantile(0.95)
    y_new     = (treo_vals / p95_new).clip(0, 1).fillna(0).values

    return X_new.values, y_new

def retrain_all_models(bundle, X_combined, y_combined):
    """
    Retrain all base models + meta-learner on combined data.
    Returns updated bundle dict.
    """
    from sklearn.preprocessing import RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.linear_model import Ridge

    n = len(X_combined)
    print(f"  Combined training set: {n} holes")

    # Re-fit scaler + PCA on combined data
    scaler  = RobustScaler()
    X_s     = scaler.fit_transform(X_combined)
    n_pca   = min(15, X_s.shape[1], n - 2)
    pca     = PCA(n_components=n_pca, random_state=42)
    X_p     = pca.fit_transform(X_s)
    kf      = KFold(n_splits=min(5, n//5), shuffle=True, random_state=42)

    new_models = {}
    new_preds  = {}
    new_scores = {}

    # Retrain each base model
    for name, model_obj in bundle["models"].items():
        try:
            from sklearn.base import clone
            m = clone(model_obj) if hasattr(model_obj, 'fit') else model_obj
            # For pipelines, use raw X; for PCA-pretransformed, use X_p
            X_fit = X_combined if hasattr(m, 'steps') else X_p
            cv_p  = cross_val_predict(m, X_fit, y_combined, cv=kf)
            m.fit(X_fit, y_combined)
            r2 = r2_score(y_combined, cv_p)
            new_models[name] = m
            new_preds[name]  = cv_p
            new_scores[name] = r2
            print(f"  {name.upper():<8} CV R² = {r2:.4f}")
        except Exception as e:
            print(f"  {name.upper():<8} retrain failed: {e} — keeping old model")
            new_models[name] = bundle["models"][name]
            new_preds[name]  = bundle["meta"].predict(
                np.column_stack([bundle["meta_info"].get("meta_weights",{}).get(k,0)
                                 for k in sorted(new_preds)])
            ) if new_preds else np.zeros(n)

    # Retrain meta-learner
    col_order = sorted(new_preds.keys())
    meta_X    = np.column_stack([new_preds[k] for k in col_order])
    meta      = Ridge(alpha=1.0)
    meta_cv   = cross_val_predict(meta, meta_X, y_combined, cv=kf).clip(0, 1)
    meta.fit(meta_X, y_combined)
    meta_r2   = r2_score(y_combined, meta_cv)
    rmse      = np.sqrt(mean_squared_error(y_combined, meta_cv))
    print(f"  ENSEMBLE  CV R² = {meta_r2:.4f}  RMSE = {rmse:.4f}")

    # Binary metrics
    thr     = np.percentile(y_combined, 70)
    yb      = (y_combined >= thr).astype(int)
    try:
        roc = roc_auc_score(yb, meta_cv)
        ap  = average_precision_score(yb, meta_cv)
    except Exception:
        roc = ap = 0.0
    print(f"  ROC AUC = {roc:.4f}  AP = {ap:.4f}")

    return new_models, meta, scaler, pca, meta_r2, rmse, roc, ap

def save_new_bundle(bundle, new_models, meta, scaler, pca,
                    X_combined, y_combined, new_deposits,
                    new_metrics, out_dir):
    """Save versioned bundle — never overwrites previous."""
    ts      = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    p       = Path(out_dir) / f"ree_model_bundle_{ts}.joblib"
    old     = bundle["meta_info"]
    new_info = {
        "version":          f"v{ts}",
        "trained_date":     datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "deposits":         new_deposits,
        "n_holes_labelled": int(len(y_combined)),
        "cv_r2":            float(new_metrics["r2"]),
        "roc_auc":          float(new_metrics["roc"]),
        "rmse":             float(new_metrics["rmse"]),
        "model_names":      sorted(new_models.keys()),
        "meta_weights":     {k: float(v) for k,v in zip(sorted(new_models), meta.coef_)},
        "feature_count":    bundle["meta_info"]["feature_count"],
        "previous_version": old.get("version","?"),
        "delta_r2":         float(new_metrics["r2"] - old.get("cv_r2", 0)),
        "delta_holes":      int(len(y_combined)) - int(old.get("n_holes_labelled", 0)),
        "notes":            f"Incremental retrain. Added: {set(new_deposits) - set(old.get('deposits',[]))}",
    }
    new_bundle = {
        "models":        new_models,
        "meta":          meta,
        "scaler":        scaler,
        "pca":           pca,
        "scaler_shared": None,
        "pca_shared":    None,
        "feat_cols":     bundle["feat_cols"],
        "p95_treo":      bundle["p95_treo"],
        "X_train":       X_combined,
        "y_train":       y_combined,
        "X_hash":        hashlib.sha256(X_combined.tobytes()).hexdigest()[:16],
        "meta_info":     new_info,
    }
    joblib.dump(new_bundle, str(p), compress=3)
    mb = p.stat().st_size / 1024 / 1024

    print()
    print("="*55)
    print("  MODEL UPDATE COMPLETE")
    print("="*55)
    print(f"  Saved:         {p.name}  ({mb:.1f} MB)")
    print(f"  Deposits now:  {new_deposits}")
    print(f"  Training holes:{len(y_combined)}")
    print(f"  CV R²:         {new_metrics['r2']:.4f}  "
          f"({'↑' if new_metrics['r2'] > old.get('cv_r2',0) else '↓'}"
          f"{abs(new_metrics['r2'] - old.get('cv_r2',0)):.4f})")
    print(f"  ROC AUC:       {new_metrics['roc']:.4f}")
    print("="*55)
    return p

# ── MAIN ──────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Incrementally retrain the REE prospectivity model with new data"
    )
    parser.add_argument("--mode", choices=["update_deposit","new_deposit"],
                        default="update_deposit",
                        help="update_deposit: new holes for existing deposit  "
                             "new_deposit: add a completely new mine")
    parser.add_argument("--deposit", required=True,
                        help="Deposit name, e.g. mount_weld or browns_range")
    parser.add_argument("--collar",  required=True, help="Path to collar CSV")
    parser.add_argument("--assay",   required=True, help="Path to assay pivoted CSV")
    parser.add_argument("--geophys", default=None,
                        help="Optional folder with geophysics TIFs for new deposit")
    parser.add_argument("--model_dir", default=MODEL_DIR,
                        help="Folder containing ree_model_bundle.joblib")
    args = parser.parse_args()

    print()
    print("="*55)
    print("  REE ENGINE — INCREMENTAL RETRAIN")
    print("="*55)
    print(f"  Mode:    {args.mode}")
    print(f"  Deposit: {args.deposit}")
    print()

    # 1. Load existing bundle
    print("Loading existing model bundle...")
    bundle_path = find_latest_bundle(args.model_dir)
    bundle      = load_bundle(bundle_path)
    print()

    # 2. Process new data
    print("Processing new drillhole data...")
    X_new, y_new = process_new_collar_assay(
        args.collar, args.assay, bundle["feat_cols"]
    )
    if X_new is None:
        sys.exit(1)

    # 3. Combine with existing training data
    X_old = bundle["X_train"]
    y_old = bundle["y_train"]
    print(f"  Existing training data: {len(y_old)} holes")
    print(f"  New data:               {len(y_new)} holes")
    X_combined = np.vstack([X_old, X_new])
    y_combined = np.concatenate([y_old, y_new])
    print(f"  Combined:               {len(y_combined)} holes")
    print()

    # 4. Retrain
    print("Retraining all models on combined data...")
    new_models, meta, scaler, pca, r2, rmse, roc, ap = retrain_all_models(
        bundle, X_combined, y_combined
    )

    # 5. Update deposit list
    old_deposits  = bundle["meta_info"].get("deposits", [])
    if args.deposit not in old_deposits:
        new_deposits = old_deposits + [args.deposit]
    else:
        new_deposits = old_deposits  # same deposit, more holes

    # 6. Save
    save_new_bundle(
        bundle, new_models, meta, scaler, pca,
        X_combined, y_combined, new_deposits,
        {"r2": r2, "rmse": rmse, "roc": roc, "ap": ap},
        args.model_dir
    )

    print()
    print("  Next step: run step3_map.py — it will auto-load the new bundle")
    input("  Press Enter to close...")

if __name__ == "__main__":
    main()

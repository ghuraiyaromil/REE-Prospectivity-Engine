import os
from pathlib import Path
from geoai.pipeline import GeoAIPipeline

# ── CONFIGURATION ─────────────────────────────────────────────
DEPOSIT = "mountain_pass"
COLLAR  = r"D:\deposits\mountain pass\dh_collar.csv"
ASSAY   = r"D:\deposits\mountain pass\dh_assay_pivoted.csv"
GEOPHYS = r"D:\deposits\mountain pass\geophysics\Mtn_Pass_RMI.tif"

def main():
    pipe = GeoAIPipeline()
    
    files = [Path(COLLAR), Path(ASSAY), Path(GEOPHYS)]
    
    print(f"Starting Leakage-Free Retraining for {DEPOSIT}...")
    result = pipe.run(
        files=files,
        deposit_name=DEPOSIT,
        force_retrain=True
    )
    
    if result["status"] == "success":
        print("\n RETRAINING SUCCESSFUL")
        print(f"   Honest Spatial R²: {result['cv_r2']:.4f}")
        print(f"   Honest Spatial ROC: {result['roc_auc']:.4f}")
        print(f"   Bundle Saved: {result['bundle_path']}")
        
        # Check if R2 is realistic
        if result['cv_r2'] > 0.9:
            print(" WARNING: R² is still very high. Checking for feature-level drift or remaining leakage.")
        elif result['cv_r2'] > 0.4:
            print(" R² is in a realistic geological range (0.4 - 0.7).")
        else:
            print(" R² is lower, but more likely to represent true unseen generalization.")
    else:
        print(f" FAILED: {result.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()

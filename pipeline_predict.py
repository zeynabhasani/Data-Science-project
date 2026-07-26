"""
Phase 3 - Prediction Pipeline (end-to-end automation)
=======================================================
Runs, in order: data loading -> preprocessing (reusing saved encoder/scaler)
-> feature engineering (reusing saved TF-IDF vectorizer + medians) -> load
trained models (NOT retrained here) -> generate predictions -> save
predictions back to the database. A single command scores the whole
dataset from scratch:

    python pipeline_predict.py

Requires pipeline_train.py to have been run at least once (so that
outputs/models/ contains the trained models + fitted preprocessing
artifacts).
"""
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"


def run(script: str):
    print(f"\n{'=' * 55}")
    print(f"  ▶ Running: {script}")
    print(f"{'=' * 55}")
    subprocess.run([sys.executable, str(SCRIPTS / script)], check=True)


if __name__ == "__main__":
    print(" Books to Scrape — PREDICTION Pipeline (Phase 3)")
    # save_predictions.py internally runs: load_data -> preprocess_inference
    # -> feature_engineering_inference -> load models -> predict -> save to DB
    run("save_predictions.py")
    print("\n✅ Prediction pipeline completed successfully.")
    print("   - Predictions saved → database/dataset.db (table: predictions)")
    print("   - Local CSV copy   → outputs/predictions.csv")

"""
Phase 3 - Training Pipeline (end-to-end automation)
=====================================================
Runs, in order: data loading -> preprocessing -> feature engineering ->
data splitting -> model training/selection/evaluation -> model saved to
outputs/models/. A single command trains the whole thing from scratch:

    python pipeline_train.py
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
    print(" Books to Scrape — TRAINING Pipeline (Phase 3)")
    run("import_to_db.py")          # CSV -> SQLite
    # split_data.py internally runs: load_data -> preprocess -> feature_engineering -> split
    run("split_data.py")            # Data Loading + Preprocessing + Feature Engineering + Splitting
    run("train_model.py")           # Model Selection + Training + Evaluation
    print("\n✅ Training pipeline completed successfully.")
    print("   - Trained models  → outputs/models/best_classifier.pkl, best_regressor.pkl")
    print("   - Fitted artifacts → outputs/models/category_encoder.pkl, scaler.pkl, feature_stats.json")
    print("   - Evaluation report → outputs/evaluation_report.json")

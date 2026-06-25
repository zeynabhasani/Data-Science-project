import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).parent / "scripts"

def run(script: str):
    print(f"\n{'='*55}")
    print(f"  ▶ Running: {script}")
    print(f"{'='*55}")
    result = subprocess.run([sys.executable, str(SCRIPTS / script)], check=True)
    return result

if __name__ == "__main__":
    print(" Books to Scrape — Data Science Pipeline (Phase 2)")
    run("import_to_db.py")
    run("load_data.py")
    run("preprocess.py")
    run("feature_engineering.py")
    print("\n✅ Pipeline completed successfully.")
    print("   Outputs saved in: outputs/")
    print("   - books_preprocessed.csv")
    print("   - books_features.csv")
    print("   - books_features.pkl")
    print("   - tfidf_vectorizer.pkl")
    print("   - eda_report.json")

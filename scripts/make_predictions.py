"""
Phase 3 - Section 3.2 (step 5): Prediction Pipeline - Modeling + Prediction
=============================================================================
Loads books from the database (this plays the role of "new/test data" for
this project, since Books-to-Scrape does not receive new records), applies
the EXACT SAME preprocessing + feature engineering used at training time
(reusing the saved encoder/scaler/TF-IDF vectorizer/medians -- transform
only, no re-fitting), loads the trained models saved by train_model.py, and
produces predictions. The trained model is NOT retrained here.

Output: a DataFrame with book_id + predicted rating_class (and probability)
+ predicted rating, ready to be persisted by save_predictions.py.
"""
import sys
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_books
from preprocess import preprocess_inference
from feature_engineering import engineer_features_inference
from split_data import get_feature_columns

BASE_DIR = Path(__file__).parent.parent
MODEL_DIR = BASE_DIR / "outputs" / "models"


def generate_predictions(df_raw: pd.DataFrame = None) -> pd.DataFrame:
    if df_raw is None:
        df_raw = load_books()

    print("\n" + "=" * 55)
    print("  Prediction Pipeline - preprocessing + feature engineering")
    print("=" * 55)
    df_clean = preprocess_inference(df_raw)
    df_feat = engineer_features_inference(df_clean)

    feature_cols = get_feature_columns(df_feat)
    # Guard against any train/inference column mismatch (e.g. an unseen
    # price bucket or missing TF-IDF term column) by reindexing to the
    # exact training feature set, filling any gaps with 0.
    X = df_feat.reindex(columns=feature_cols, fill_value=0)

    clf = joblib.load(MODEL_DIR / "best_classifier.pkl")
    reg = joblib.load(MODEL_DIR / "best_regressor.pkl")
    print(f"\n  Loaded models: {type(clf.named_steps['clf'] if hasattr(clf, 'named_steps') else clf).__name__} (classifier), "
          f"{type(reg.named_steps['reg'] if hasattr(reg, 'named_steps') else reg).__name__} (regressor)")

    pred_class = clf.predict(X)
    pred_proba = clf.predict_proba(X)[:, 1] if hasattr(clf, "predict_proba") else None
    pred_rating = reg.predict(X)

    result = pd.DataFrame({
        "book_id": df_feat["book_id"].values,
        "predicted_rating_class": pred_class,
        "predicted_rating_class_proba": pred_proba if pred_proba is not None else float("nan"),
        "predicted_rating": pred_rating.round(2),
    })
    print(f"\n✅ Generated predictions for {len(result)} books.")
    print(result.head(10).to_string(index=False))
    return result


if __name__ == "__main__":
    preds = generate_predictions()
    out_path = BASE_DIR / "outputs" / "predictions.csv"
    preds.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved local copy → {out_path}")

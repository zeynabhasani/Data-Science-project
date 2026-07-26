import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
import sys, json, pickle

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_books

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR = OUTPUT_DIR / "models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

def run_eda(df):
    print("\n" + "="*55)
    print("  EDA — Exploratory Data Analysis")
    print("="*55)
    print(f"\n Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    missing = df.isnull().sum()
    print("\n Missing values:")
    print(missing[missing > 0].to_string() if missing.sum() > 0 else "  2 missing descriptions — will be imputed.")

    print("\n Numeric summary:")
    print(df[["price_incl_tax","stock_count","rating"]].describe().round(2).to_string())

    print("\n Rating distribution (TARGET):")
    print(df["rating"].value_counts().sort_index().to_string())

    print("\n Top 10 categories:")
    print(df["category"].value_counts().head(10).to_string())

    print("\n Price stats:")
    print(f"  mean=£{df['price_incl_tax'].mean():.2f}  std=£{df['price_incl_tax'].std():.2f}  min=£{df['price_incl_tax'].min():.2f}  max=£{df['price_incl_tax'].max():.2f}")

    print("\n Description length:")
    dl = df["description"].str.len()
    print(f"  min={dl.min()}  max={dl.max()}  mean={dl.mean():.0f}  median={dl.median():.0f}")

    print("\n Price vs Rating correlation:")
    print(f"  Pearson r = {df['price_incl_tax'].corr(df['rating']):.4f}")

    outliers = df[df["price_incl_tax"] > df["price_incl_tax"].mean() + 2*df["price_incl_tax"].std()]
    print(f"\n  Price outliers (>mean+2σ): {len(outliers)} books")

    eda = {
        "shape": list(df.shape),
        "missing": missing.to_dict(),
        "rating_distribution": df["rating"].value_counts().sort_index().to_dict(),
        "price_mean": round(float(df["price_incl_tax"].mean()), 2),
        "price_std": round(float(df["price_incl_tax"].std()), 2),
        "category_count": int(df["category"].nunique()),
        "outlier_count": int(len(outliers)),
        "price_rating_corr": round(float(df["price_incl_tax"].corr(df["rating"])), 4),
    }
    with open(OUTPUT_DIR / "eda_report.json", "w") as f:
        json.dump(eda, f, indent=2)
    print("\n✅ EDA saved → outputs/eda_report.json")

def preprocess(df):
    print("\n" + "="*55)
    print("  Preprocessing")
    print("="*55)
    df = df.copy()

    missing_desc = df["description"].isnull().sum()
    df["description"] = df["description"].fillna("")
    print(f"  Imputed {missing_desc} missing descriptions with empty string.")

    drop_cols = []
    for col in ["tax", "price_excl_tax", "num_reviews", "product_type",
                "availability_text", "upc", "image_url", "page_url",
                "crawled_at", "category_id"]:
        if col in df.columns:
            drop_cols.append(col)
    df.drop(columns=drop_cols, inplace=True)
    print(f"  Dropped: {drop_cols}")
    print(f"  Justification: tax/num_reviews=zero variance | price_excl_tax=duplicate | rest=non-informative metadata")

    le = LabelEncoder()
    df["category_encoded"] = le.fit_transform(df["category"])
    print(f"  Label-encoded 'category' → {df['category'].nunique()} classes")

    scaler = MinMaxScaler()
    scale_cols = ["price_incl_tax", "stock_count"]
    df[[f"{c}_scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])
    print(f"  MinMax-scaled: {scale_cols}")

    # Persist the fitted encoder/scaler so the PREDICTION pipeline can apply
    # the exact same (already-fitted) transformation to new/test data instead
    # of re-fitting on it (which would cause train/inference inconsistency).
    with open(ARTIFACT_DIR / "category_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    with open(ARTIFACT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  ✅ Saved fitted encoder/scaler → outputs/models/category_encoder.pkl, scaler.pkl")

    df["rating_class"] = (df["rating"] >= 3).astype(int)
    print(f"  Target 'rating' (1-5): regression")
    print(f"  Target 'rating_class' (0/1, ≥3=1): classification | dist={df['rating_class'].value_counts().to_dict()}")

    out = OUTPUT_DIR / "books_preprocessed.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved → {out}  ({len(df)} rows × {len(df.columns)} cols)")
    return df

def preprocess_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Prediction-pipeline version of preprocess(): applies the SAME cleaning
    steps but re-uses the encoder/scaler fitted during training (loaded from
    outputs/models/) instead of fitting new ones. Falls back to fitting on
    the fly (with a warning) if the artifacts are not found, e.g. the first
    time this is run before any training has happened.
    """
    df = df.copy()
    df["description"] = df["description"].fillna("")

    drop_cols = [c for c in ["tax", "price_excl_tax", "num_reviews", "product_type",
                              "availability_text", "upc", "image_url", "page_url",
                              "crawled_at", "category_id"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    encoder_path = ARTIFACT_DIR / "category_encoder.pkl"
    scaler_path = ARTIFACT_DIR / "scaler.pkl"

    if encoder_path.exists() and scaler_path.exists():
        with open(encoder_path, "rb") as f:
            le = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
        # Categories unseen during training fall back to -1 (unknown class)
        known = set(le.classes_)
        df["category_encoded"] = df["category"].apply(
            lambda c: le.transform([c])[0] if c in known else -1
        )
        scale_cols = ["price_incl_tax", "stock_count"]
        df[[f"{c}_scaled" for c in scale_cols]] = scaler.transform(df[scale_cols])
        print("  ✅ Reused fitted encoder/scaler from outputs/models/ (no re-fit on inference data)")
    else:
        print("  ⚠️  No saved encoder/scaler found — fitting on this data as a fallback "
              "(run the training pipeline first for consistent transformations).")
        le = LabelEncoder()
        df["category_encoded"] = le.fit_transform(df["category"])
        scaler = MinMaxScaler()
        scale_cols = ["price_incl_tax", "stock_count"]
        df[[f"{c}_scaled" for c in scale_cols]] = scaler.fit_transform(df[scale_cols])

    df["rating_class"] = (df["rating"] >= 3).astype(int) if "rating" in df.columns else np.nan
    return df


if __name__ == "__main__":
    df = load_books()
    run_eda(df)
    df_clean = preprocess(df)
    print("\nFinal columns:", df_clean.columns.tolist())

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from pathlib import Path
import sys, json

sys.path.insert(0, str(Path(__file__).parent))
from load_data import load_books

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

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

    df["rating_class"] = (df["rating"] >= 3).astype(int)
    print(f"  Target 'rating' (1-5): regression")
    print(f"  Target 'rating_class' (0/1, ≥3=1): classification | dist={df['rating_class'].value_counts().to_dict()}")

    out = OUTPUT_DIR / "books_preprocessed.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n✅ Saved → {out}  ({len(df)} rows × {len(df.columns)} cols)")
    return df

if __name__ == "__main__":
    df = load_books()
    run_eda(df)
    df_clean = preprocess(df)
    print("\nFinal columns:", df_clean.columns.tolist())

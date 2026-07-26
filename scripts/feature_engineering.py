import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle, sys, json

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess, preprocess_inference
from load_data import load_books

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
ARTIFACT_DIR = OUTPUT_DIR / "models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
STATS_PATH = ARTIFACT_DIR / "feature_stats.json"

def engineer_features(df):
    print("\n" + "="*55)
    print("  Feature Engineering")
    print("="*55)
    df = df.copy()

    df["desc_length"]       = df["description"].str.len()
    df["desc_word_count"]   = df["description"].str.split().str.len()
    df["price_per_rating"]  = (df["price_incl_tax"] / df["rating"]).round(2)
    df["stock_price_ratio"] = (df["stock_count"] / df["price_incl_tax"]).round(4)
    price_median = df["price_incl_tax"].median()
    stock_median = df["stock_count"].median()
    df["is_expensive"]      = (df["price_incl_tax"] > price_median).astype(int)
    df["is_high_stock"]     = (df["stock_count"] > stock_median).astype(int)
    print("  ✅ Numeric features: desc_length, desc_word_count, price_per_rating, stock_price_ratio, is_expensive, is_high_stock")

    with open(STATS_PATH, "w") as f:
        json.dump({"price_median": float(price_median), "stock_median": float(stock_median)}, f, indent=2)
    print(f"  ✅ Saved training-time thresholds → outputs/models/feature_stats.json")

    df["price_bucket"] = pd.cut(
        df["price_incl_tax"],
        bins=[0, 20, 35, 50, 60],
        labels=["cheap", "mid", "expensive", "premium"]
    ).astype(str)
    df = pd.get_dummies(df, columns=["price_bucket"], prefix="bucket")
    print("  ✅ price_bucket → one-hot (cheap/mid/expensive/premium)")

    # ─── ۳. TF-IDF روی description (تارگت: rating) ───
    print("\n  Running TF-IDF on descriptions (100 features)...")
    tfidf = TfidfVectorizer(
        max_features=100,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        sublinear_tf=True
    )
    tfidf_matrix = tfidf.fit_transform(df["description"].fillna(""))
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
        index=df.index
    )
    df = pd.concat([df, tfidf_df], axis=1)
    print(f"  ✅ TF-IDF: {tfidf_df.shape[1]} features (unigrams+bigrams, min_df=3, sublinear_tf)")

    with open(OUTPUT_DIR / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    print("  ✅ Vectorizer saved → outputs/tfidf_vectorizer.pkl")

    numeric_f = [c for c in df.columns if df[c].dtype in [float, int]
                 and "tfidf" not in c and "bucket" not in c
                 and c not in ["rating","rating_class","category_encoded"]]
    tfidf_f   = [c for c in df.columns if "tfidf" in c]
    bucket_f  = [c for c in df.columns if "bucket" in c]

    print(f"\n Final feature summary:")
    print(f"  Numeric features ({len(numeric_f)}): {numeric_f}")
    print(f"  TF-IDF features:  {len(tfidf_f)}")
    print(f"  One-hot (bucket): {len(bucket_f)} → {bucket_f}")
    print(f"  TARGET (regression):     rating (1-5)")
    print(f"  TARGET (classification): rating_class (0/1)")

    final_path = OUTPUT_DIR / "books_features.csv"
    df.to_csv(final_path, index=False, encoding="utf-8-sig")
    df.to_pickle(OUTPUT_DIR / "books_features.pkl")
    print(f"\n✅ Final dataset saved:")
    print(f"   CSV    → {final_path}")
    print(f"   Pickle → outputs/books_features.pkl")
    print(f"   Shape  → {df.shape[0]} rows × {df.shape[1]} columns")
    return df

def engineer_features_inference(df: pd.DataFrame) -> pd.DataFrame:
    """Prediction-pipeline version of engineer_features(): reuses the TF-IDF
    vectorizer and price/stock medians fitted during training (transform
    only, no re-fit), so features generated for new/test data are on the
    exact same scale/vocabulary the model was trained on.
    """
    df = df.copy()
    df["desc_length"]       = df["description"].str.len()
    df["desc_word_count"]   = df["description"].str.split().str.len()
    df["price_per_rating"]  = (df["price_incl_tax"] / df["rating"]).round(2) if "rating" in df.columns else np.nan
    df["stock_price_ratio"] = (df["stock_count"] / df["price_incl_tax"]).round(4)

    if STATS_PATH.exists():
        with open(STATS_PATH) as f:
            stats = json.load(f)
        price_median, stock_median = stats["price_median"], stats["stock_median"]
    else:
        price_median, stock_median = df["price_incl_tax"].median(), df["stock_count"].median()

    df["is_expensive"]  = (df["price_incl_tax"] > price_median).astype(int)
    df["is_high_stock"] = (df["stock_count"] > stock_median).astype(int)

    df["price_bucket"] = pd.cut(
        df["price_incl_tax"], bins=[0, 20, 35, 50, 60],
        labels=["cheap", "mid", "expensive", "premium"]
    ).astype(str)
    df = pd.get_dummies(df, columns=["price_bucket"], prefix="bucket")
    for col in ["bucket_cheap", "bucket_mid", "bucket_expensive", "bucket_premium"]:
        if col not in df.columns:
            df[col] = False

    tfidf_path = OUTPUT_DIR / "tfidf_vectorizer.pkl"
    with open(tfidf_path, "rb") as f:
        tfidf = pickle.load(f)
    tfidf_matrix = tfidf.transform(df["description"].fillna(""))
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=[f"tfidf_{w}" for w in tfidf.get_feature_names_out()],
        index=df.index
    )
    df = pd.concat([df, tfidf_df], axis=1)
    print(f"  ✅ Reused fitted TF-IDF vectorizer + training medians (no re-fit on inference data)")
    return df


if __name__ == "__main__":
    raw   = load_books()
    clean = preprocess(raw)
    final = engineer_features(clean)

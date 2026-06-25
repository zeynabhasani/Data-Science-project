import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path
import pickle, sys

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import preprocess
from load_data import load_books

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

def engineer_features(df):
    print("\n" + "="*55)
    print("  Feature Engineering")
    print("="*55)
    df = df.copy()

    df["desc_length"]       = df["description"].str.len()
    df["desc_word_count"]   = df["description"].str.split().str.len()
    df["price_per_rating"]  = (df["price_incl_tax"] / df["rating"]).round(2)
    df["stock_price_ratio"] = (df["stock_count"] / df["price_incl_tax"]).round(4)
    df["is_expensive"]      = (df["price_incl_tax"] > df["price_incl_tax"].median()).astype(int)
    df["is_high_stock"]     = (df["stock_count"] > df["stock_count"].median()).astype(int)
    print("  ✅ Numeric features: desc_length, desc_word_count, price_per_rating, stock_price_ratio, is_expensive, is_high_stock")

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

if __name__ == "__main__":
    raw   = load_books()
    clean = preprocess(raw)
    final = engineer_features(clean)

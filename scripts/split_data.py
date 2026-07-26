"""
Phase 3 - Section 1.2: Data Splitting
Splits the final engineered feature set (outputs/books_features.csv) into
train / validation / test sets for both the regression target (rating) and
the classification target (rating_class).

Split ratio: 70% train / 15% validation / 15% test
Stratification is applied on `rating_class` for the classification split so
that the class balance (~58/42) is preserved in every subset.
"""
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
from feature_engineering import engineer_features
from preprocess import preprocess
from load_data import load_books

OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
SPLIT_DIR = OUTPUT_DIR / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# Columns that must NEVER be used as model input.
# - identifiers / free text: not usable directly by the models we test
# - `category`, `description`: already encoded as category_encoded / TF-IDF
# - `price_per_rating`: engineered in Phase 2 as price_incl_tax / rating --
#   this is DATA LEAKAGE (it directly encodes the target) and is dropped
#   here before any modeling step.
NON_FEATURE_COLS = [
    "book_id", "title", "description", "category",
    "rating", "rating_class", "price_per_rating",
]


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLS]


def split_and_save(df: pd.DataFrame, test_size=0.15, val_size=0.15, random_state=42):
    feature_cols = get_feature_columns(df)
    print(f"  Using {len(feature_cols)} feature columns (dropped leakage/id/text columns: {NON_FEATURE_COLS})")

    # First split off the test set, then split remaining into train/val.
    train_val, test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["rating_class"]
    )
    val_ratio_within_train_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_ratio_within_train_val, random_state=random_state,
        stratify=train_val["rating_class"]
    )

    for name, part in [("train", train), ("val", val), ("test", test)]:
        path = SPLIT_DIR / f"{name}.csv"
        part.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"  {name:5s}: {len(part):4d} rows -> {path}")

    print("\n  Class balance (rating_class) per split:")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        dist = part["rating_class"].value_counts(normalize=True).round(3).to_dict()
        print(f"    {name:5s}: {dist}")

    return train, val, test, feature_cols


if __name__ == "__main__":
    print("=" * 55)
    print("  Data Splitting (train / val / test)")
    print("=" * 55)
    raw = load_books()
    clean = preprocess(raw)
    final = engineer_features(clean)
    split_and_save(final)
    print("\n✅ Splits saved under outputs/splits/")

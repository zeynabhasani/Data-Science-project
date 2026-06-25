import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from database_connection import get_connection

def load_books() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("""
            SELECT b.*, c.name AS category
            FROM books b
            JOIN categories c ON b.category_id = c.category_id
        """, conn)
    print(f"✅ Loaded {len(df)} books from database.")
    return df

def load_categories() -> pd.DataFrame:
    with get_connection() as conn:
        df = pd.read_sql_query("SELECT * FROM categories", conn)
    return df

if __name__ == "__main__":
    df = load_books()
    print(df[["title","category","price_incl_tax","rating"]].head(10).to_string())

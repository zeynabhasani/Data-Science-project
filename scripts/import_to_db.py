import pandas as pd
import sqlite3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from database_connection import get_connection, DB_PATH

RAW_CSV = Path(__file__).parent.parent / "database" / "books_raw.csv"

def create_schema(conn):
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT UNIQUE NOT NULL
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id        TEXT PRIMARY KEY,
            title          TEXT NOT NULL,
            price_incl_tax REAL,
            price_excl_tax REAL,
            tax            REAL,
            stock_count    INTEGER,
            rating         INTEGER,
            category_id    INTEGER REFERENCES categories(category_id),
            upc            TEXT,
            product_type   TEXT,
            num_reviews    INTEGER,
            description    TEXT,
            image_url      TEXT,
            page_url       TEXT,
            crawled_at     TEXT
        )
    """)
    conn.commit()
    print("✅ Schema created.")

def import_data(conn):
    df = pd.read_csv(RAW_CSV)
    cursor = conn.cursor()

    for cat in df["category"].dropna().unique():
        cursor.execute("INSERT OR IGNORE INTO categories (name) VALUES (?)", (cat,))
    conn.commit()

    cat_map = {r[0]: r[1] for r in cursor.execute("SELECT name, category_id FROM categories")}
    df["category_id"] = df["category"].map(cat_map)

    cols = ["book_id","title","price_incl_tax","price_excl_tax","tax",
            "stock_count","rating","category_id","upc","product_type",
            "num_reviews","description","image_url","page_url","crawled_at"]
    df[cols].to_sql("books", conn, if_exists="replace", index=False)
    print(f"✅ Imported {len(df)} books | {df['category'].nunique()} categories")

    print("\n Top 5 expensive books:")
    for r in conn.execute("""
        SELECT b.title, b.price_incl_tax, b.rating, c.name
        FROM books b JOIN categories c ON b.category_id=c.category_id
        ORDER BY b.price_incl_tax DESC LIMIT 5
    """).fetchall():
        print(f"  {r[0][:40]:<40} £{r[1]:.2f}  ⭐{r[2]}  [{r[3]}]")

    print("\n Books per category (top 10):")
    for r in conn.execute("""
        SELECT c.name, COUNT(*) as cnt
        FROM books b JOIN categories c ON b.category_id=c.category_id
        GROUP BY c.name ORDER BY cnt DESC LIMIT 10
    """).fetchall():
        print(f"  {r[0]:<25} {r[1]}")

    print("\n Average rating per category (top 10):")
    for r in conn.execute("""
        SELECT c.name, ROUND(AVG(b.rating),2) as avg_rating, COUNT(*) as cnt
        FROM books b JOIN categories c ON b.category_id=c.category_id
        GROUP BY c.name HAVING cnt >= 5
        ORDER BY avg_rating DESC LIMIT 10
    """).fetchall():
        print(f"  {r[0]:<25} avg={r[1]}  n={r[2]}")

if __name__ == "__main__":
    print(f" DB: {DB_PATH}")
    with get_connection() as conn:
        create_schema(conn)
        import_data(conn)

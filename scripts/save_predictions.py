"""
Phase 3 - Section 3.2 (step 5): Prediction Pipeline - Saving predictions to DB
================================================================================
Takes the DataFrame produced by make_predictions.py and writes it back to
the project's SQLite database (database/dataset.db) in a new `predictions`
table, so results are stored and accessible for later analysis or
downstream/production use -- satisfying the "must save final predictions
back to the database" requirement.
"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from database_connection import get_connection
from make_predictions import generate_predictions

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    book_id TEXT PRIMARY KEY,
    predicted_rating_class INTEGER,
    predicted_rating_class_proba REAL,
    predicted_rating REAL,
    predicted_at TEXT,
    FOREIGN KEY (book_id) REFERENCES books (book_id)
);
"""


def save_predictions(preds: pd.DataFrame):
    preds = preds.copy()
    preds["predicted_at"] = datetime.now(timezone.utc).isoformat()

    with get_connection() as conn:
        conn.execute(CREATE_TABLE_SQL)
        conn.commit()
        # Upsert-style replace: predictions table is a snapshot of the most
        # recent scoring run, keyed by book_id.
        preds.to_sql("predictions", conn, if_exists="replace", index=False)
        conn.commit()

        n = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        print(f"✅ Saved {n} predictions to database/dataset.db -> table `predictions`")

        print("\nSample rows from `predictions` table:")
        sample = pd.read_sql_query("SELECT * FROM predictions LIMIT 5", conn)
        print(sample.to_string(index=False))


if __name__ == "__main__":
    preds = generate_predictions()
    save_predictions(preds)
    csv_path = Path(__file__).parent.parent / "outputs" / "predictions.csv"
    preds.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n✅ Local CSV snapshot → {csv_path}")

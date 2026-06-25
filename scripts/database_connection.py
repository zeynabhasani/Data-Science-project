import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "database" / "dataset.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_engine():
    from sqlalchemy import create_engine
    return create_engine(f"sqlite:///{DB_PATH}")

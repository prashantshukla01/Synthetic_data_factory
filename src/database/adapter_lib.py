import sqlite3
import datetime
import os

DB_PATH = "data/adapters.db"

def init_db():
    """Initializes the local adapter registry."""
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS adapters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            base_model TEXT NOT NULL,
            path TEXT NOT NULL,
            date_trained TEXT,
            accuracy_gain REAL
        )
    ''')
    conn.commit()
    conn.close()

def register_adapter(name, base_model, path, accuracy_gain=0.0):
    """Logs a new fine-tuned model into the library."""
    # Ensure database is initialized
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    cursor.execute(
        "INSERT INTO adapters (name, base_model, path, date_trained, accuracy_gain) VALUES (?, ?, ?, ?, ?)",
        (name, base_model, path, date, accuracy_gain)
    )
    conn.commit()
    conn.close()

def list_adapters():
    """Returns all saved adapters for the UI dropdown."""
    # Ensure database is initialized
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, base_model, path FROM adapters")
    data = cursor.fetchall()
    conn.close()
    return data
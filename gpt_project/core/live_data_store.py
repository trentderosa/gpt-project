import json
import sqlite3
import os
from pathlib import Path

from .config import BASE_DIR


def _default_db_path() -> Path:
    configured = (os.getenv("DB_PATH") or "").strip()
    if configured:
        return Path(configured)
    if (os.getenv("RENDER") or "").strip().lower() == "true":
        return Path("/var/data/gpt_project.db")
    return BASE_DIR / "gpt_project.db"


DB_PATH = _default_db_path()


class LiveDataStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS live_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_type TEXT NOT NULL,
                    source_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def upsert_snapshot(self, source_type: str, source_key: str, payload: dict) -> None:
        payload_json = json.dumps(payload, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO live_data (source_type, source_key, payload)
                VALUES (?, ?, ?)
                """,
                (source_type, source_key, payload_json),
            )
            conn.execute(
                """
                DELETE FROM live_data
                WHERE id NOT IN (
                    SELECT id
                    FROM live_data
                    WHERE source_type = ? AND source_key = ?
                    ORDER BY id DESC
                    LIMIT 1
                ) AND source_type = ? AND source_key = ?
                """,
                (source_type, source_key, source_type, source_key),
            )

    def get_latest(self, source_type: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT source_key, payload, created_at
                FROM live_data
                WHERE source_type = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (source_type, limit),
            ).fetchall()
        results: list[dict] = []
        for row in rows:
            try:
                payload = json.loads(row["payload"])
            except Exception:
                payload = {"raw": row["payload"]}
            results.append(
                {
                    "source_key": row["source_key"],
                    "payload": payload,
                    "created_at": row["created_at"],
                }
            )
        return results

import sqlite3
import uuid
import os
import json
import re
import hashlib
import hmac
import secrets
from pathlib import Path
from datetime import datetime, timedelta, timezone

from .config import BASE_DIR


DB_PATH = Path(os.getenv("DB_PATH") or (BASE_DIR / "gpt_project.db"))


class ChatStorage:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_profiles (
                    conversation_id TEXT PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS uploaded_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    extracted_text TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    plan TEXT NOT NULL DEFAULT 'free',
                    stripe_customer_id TEXT,
                    stripe_subscription_id TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at DATETIME NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_memory (
                    user_id INTEGER PRIMARY KEY,
                    profile_json TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "user_id" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")
            user_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(users)").fetchall()
            }
            if "stripe_customer_id" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN stripe_customer_id TEXT")
            if "stripe_subscription_id" not in user_columns:
                conn.execute("ALTER TABLE users ADD COLUMN stripe_subscription_id TEXT")

    def create_conversation(self, user_id: int | None = None) -> str:
        conversation_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id) VALUES (?, ?)",
                (conversation_id, user_id),
            )
        return conversation_id

    def conversation_exists(self, conversation_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return row is not None

    def conversation_owner(self, conversation_id: str) -> int | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT user_id FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        if not row:
            return None
        return row["user_id"]

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (conversation_id, role, content) VALUES (?, ?, ?)",
                (conversation_id, role, content),
            )

    def get_messages(self, conversation_id: str, limit: int | None = None) -> list[dict]:
        query = """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """
        params: list = [conversation_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)

        with self._connect() as conn:
            rows = conn.execute(query, tuple(params)).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def list_conversations(self, limit: int = 30, user_id: int | None = None) -> list[dict]:
        query = """
            SELECT
                c.id AS conversation_id,
                c.created_at AS created_at,
                (
                    SELECT MAX(m2.created_at)
                    FROM messages m2
                    WHERE m2.conversation_id = c.id
                ) AS last_message_at
            FROM conversations c
            {where_clause}
            ORDER BY COALESCE(last_message_at, c.created_at) DESC
            LIMIT ?
        """
        where_clause = ""
        params: list[object] = []
        if user_id is not None:
            where_clause = "WHERE c.user_id = ?"
            params.append(user_id)
        query = query.format(where_clause=where_clause)
        with self._connect() as conn:
            params.append(limit)
            rows = conn.execute(query, tuple(params)).fetchall()
            items: list[dict] = []
            for row in rows:
                preview = self._build_conversation_preview(conn, row["conversation_id"])
                items.append(
                    {
                        "conversation_id": row["conversation_id"],
                        "created_at": row["created_at"],
                        "last_message_at": row["last_message_at"],
                        "preview": preview,
                    }
                )
            return items

    def _build_conversation_preview(self, conn: sqlite3.Connection, conversation_id: str) -> str:
        rows = conn.execute(
            """
            SELECT role, content
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id DESC
            LIMIT 10
            """,
            (conversation_id,),
        ).fetchall()
        if not rows:
            return "New conversation"

        recent = [{"role": row["role"], "content": row["content"]} for row in reversed(rows)]
        user_messages = [m["content"] for m in recent if m["role"] == "user"]
        assistant_messages = [m["content"] for m in recent if m["role"] == "assistant"]

        latest_user = self._clean_preview_text(user_messages[-1]) if user_messages else ""
        latest_assistant = self._clean_preview_text(assistant_messages[-1]) if assistant_messages else ""

        # Use multiple turns to generate a compact summary-style title.
        if latest_user and latest_assistant:
            summary = f"{latest_user} -> {latest_assistant}"
        else:
            summary = latest_user or latest_assistant or "New conversation"
        return summary[:90]

    def _clean_preview_text(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        cleaned = re.sub(r"\s+", " ", cleaned)
        cleaned = re.sub(r"[*_`#>\[\]\(\)]+", "", cleaned)
        if len(cleaned) > 45:
            cleaned = cleaned[:45].rstrip() + "..."
        return cleaned

    def get_user_profile(self, conversation_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_profiles WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["profile_json"])
        except Exception:
            return {}

    def upsert_user_profile(self, conversation_id: str, profile: dict) -> None:
        payload = json.dumps(profile, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_profiles (conversation_id, profile_json)
                VALUES (?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (conversation_id, payload),
            )

    def add_uploaded_file(
        self,
        conversation_id: str,
        filename: str,
        media_type: str,
        extracted_text: str,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO uploaded_files (conversation_id, filename, media_type, extracted_text)
                VALUES (?, ?, ?, ?)
                """,
                (conversation_id, filename, media_type, extracted_text),
            )
            return int(cursor.lastrowid)

    def get_uploaded_files(self, conversation_id: str, limit: int = 12) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, media_type, extracted_text, created_at
                FROM uploaded_files
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "filename": row["filename"],
                "media_type": row["media_type"],
                "extracted_text": row["extracted_text"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def _hash_password(self, password: str, salt: str | None = None) -> str:
        chosen_salt = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            chosen_salt.encode("utf-8"),
            120_000,
        )
        return f"{chosen_salt}${digest.hex()}"

    def _verify_password(self, password: str, encoded: str) -> bool:
        try:
            salt, expected = encoded.split("$", 1)
        except ValueError:
            return False
        actual = self._hash_password(password, salt=salt).split("$", 1)[1]
        return hmac.compare_digest(actual, expected)

    def create_user(self, email: str, password: str) -> dict:
        normalized = email.strip().lower()
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        creator_email = (os.getenv("CREATOR_EMAIL") or "").strip().lower()
        plan = "creator" if creator_email and normalized == creator_email else "free"
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT id FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
            if exists:
                raise ValueError("Email is already registered.")
            password_hash = self._hash_password(password)
            cur = conn.execute(
                "INSERT INTO users (email, password_hash, plan) VALUES (?, ?, ?)",
                (normalized, password_hash, plan),
            )
            user_id = int(cur.lastrowid)
            row = conn.execute(
                "SELECT id, email, plan, stripe_customer_id, stripe_subscription_id, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row)

    def authenticate_user(self, email: str, password: str) -> dict | None:
        normalized = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, password_hash, plan, stripe_customer_id, stripe_subscription_id, created_at FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        if not row:
            return None
        if not self._verify_password(password, row["password_hash"]):
            return None
        return {
            "id": row["id"],
            "email": row["email"],
            "plan": row["plan"],
            "stripe_customer_id": row["stripe_customer_id"],
            "stripe_subscription_id": row["stripe_subscription_id"],
            "created_at": row["created_at"],
        }

    def create_session(self, user_id: int, ttl_days: int = 30) -> str:
        token = secrets.token_urlsafe(48)
        expires_at = datetime.now(timezone.utc) + timedelta(days=max(ttl_days, 1))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO user_sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at.strftime("%Y-%m-%d %H:%M:%S")),
            )
        return token

    def get_user_by_token(self, token: str) -> dict | None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.email, u.plan, u.created_at, s.expires_at
                       ,u.stripe_customer_id, u.stripe_subscription_id
                FROM user_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ? AND s.expires_at > ?
                """,
                (token, now),
            ).fetchone()
        if not row:
            return None
        return {
            "id": row["id"],
            "email": row["email"],
            "plan": row["plan"],
            "stripe_customer_id": row["stripe_customer_id"],
            "stripe_subscription_id": row["stripe_subscription_id"],
            "created_at": row["created_at"],
            "expires_at": row["expires_at"],
        }

    def get_user_by_id(self, user_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, email, plan, stripe_customer_id, stripe_subscription_id, created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        if not row:
            return None
        return dict(row)

    def delete_session(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM user_sessions WHERE token = ?", (token,))

    def record_usage_event(self, user_id: int, event_type: str = "chat_input") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO usage_events (user_id, event_type) VALUES (?, ?)",
                (user_id, event_type),
            )

    def usage_count_last_hour(self, user_id: int, event_type: str = "chat_input") -> int:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COUNT(1) AS cnt
                FROM usage_events
                WHERE user_id = ?
                  AND event_type = ?
                  AND created_at >= datetime('now', '-1 hour')
                """,
                (user_id, event_type),
            ).fetchone()
        return int(row["cnt"] if row else 0)

    def get_user_memory(self, user_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT profile_json FROM user_memory WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            return {}
        try:
            return json.loads(row["profile_json"])
        except Exception:
            return {}

    def upsert_user_memory(self, user_id: int, profile: dict) -> None:
        payload = json.dumps(profile, ensure_ascii=True)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_memory (user_id, profile_json)
                VALUES (?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    profile_json = excluded.profile_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (user_id, payload),
            )

    def set_user_plan_by_email(self, email: str, plan: str) -> bool:
        normalized = (email or "").strip().lower()
        if not normalized:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET plan = ? WHERE email = ?",
                (plan, normalized),
            )
            return cur.rowcount > 0

    def set_user_plan_by_id(self, user_id: int, plan: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET plan = ? WHERE id = ?",
                (plan, user_id),
            )
            return cur.rowcount > 0

    def set_user_plan_by_stripe_customer(self, stripe_customer_id: str, plan: str) -> bool:
        customer = (stripe_customer_id or "").strip()
        if not customer:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET plan = ? WHERE stripe_customer_id = ?",
                (plan, customer),
            )
            return cur.rowcount > 0

    def set_user_billing_ids(
        self,
        user_id: int,
        stripe_customer_id: str | None = None,
        stripe_subscription_id: str | None = None,
    ) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE users
                SET stripe_customer_id = COALESCE(?, stripe_customer_id),
                    stripe_subscription_id = COALESCE(?, stripe_subscription_id)
                WHERE id = ?
                """,
                (stripe_customer_id, stripe_subscription_id, user_id),
            )
            return cur.rowcount > 0

    def set_subscription_by_stripe_customer(
        self,
        stripe_customer_id: str,
        stripe_subscription_id: str,
    ) -> bool:
        customer = (stripe_customer_id or "").strip()
        subscription = (stripe_subscription_id or "").strip()
        if not customer or not subscription:
            return False
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE users SET stripe_subscription_id = ? WHERE stripe_customer_id = ?",
                (subscription, customer),
            )
            return cur.rowcount > 0

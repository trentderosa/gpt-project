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


def _default_db_path() -> Path:
    configured = (os.getenv("DB_PATH") or "").strip()
    if configured:
        return Path(configured)
    # On Render, use mounted persistent disk when available.
    if (os.getenv("RENDER") or "").strip().lower() == "true":
        return Path("/var/data/gpt_project.db")
    return BASE_DIR / "gpt_project.db"


DB_PATH = _default_db_path()


class ChatStorage:
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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_knowledge_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS password_reset_tokens (
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
                CREATE TABLE IF NOT EXISTS workspaces (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    owner_user_id INTEGER NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(owner_user_id) REFERENCES users(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS workspace_members (
                    workspace_id TEXT NOT NULL,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL DEFAULT 'member',
                    joined_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (workspace_id, user_id),
                    FOREIGN KEY(workspace_id) REFERENCES workspaces(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )
                """
            )
            # Additive migration: add workspace_id to user_knowledge_files (NULL = personal file)
            knowledge_columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(user_knowledge_files)").fetchall()
            }
            if "workspace_id" not in knowledge_columns:
                conn.execute(
                    "ALTER TABLE user_knowledge_files ADD COLUMN workspace_id TEXT REFERENCES workspaces(id)"
                )
            columns = {
                row["name"]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "user_id" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN user_id INTEGER")
            if "title" not in columns:
                conn.execute("ALTER TABLE conversations ADD COLUMN title TEXT")
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
                c.title AS title,
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
                        "title": row["title"],
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
        plan = "free"
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

    def upsert_creator_user(self, email: str, password: str) -> dict:
        """Create or update the creator account, always setting plan to 'creator'."""
        normalized = email.strip().lower()
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters.")
        password_hash = self._hash_password(password)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT id FROM users WHERE email = ?", (normalized,)
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE users SET password_hash = ?, plan = 'creator' WHERE email = ?",
                    (password_hash, normalized),
                )
                user_id = int(existing["id"])
            else:
                cur = conn.execute(
                    "INSERT INTO users (email, password_hash, plan) VALUES (?, ?, 'creator')",
                    (normalized, password_hash),
                )
                user_id = int(cur.lastrowid)
            row = conn.execute(
                "SELECT id, email, plan, stripe_customer_id, stripe_subscription_id, created_at FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
        return dict(row)

    def sync_creator_plan(self, email: str) -> dict | None:
        normalized = (email or "").strip().lower()
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
            if not row:
                return None
            conn.execute(
                "UPDATE users SET plan = 'creator' WHERE email = ?",
                (normalized,),
            )
            refreshed = conn.execute(
                "SELECT id, email, plan, stripe_customer_id, stripe_subscription_id, created_at FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        return dict(refreshed) if refreshed else None

    def authenticate_user_detailed(self, email: str, password: str) -> tuple[dict | None, str]:
        normalized = email.strip().lower()
        if not normalized:
            return None, "missing_email"
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, password_hash, plan, stripe_customer_id, stripe_subscription_id, created_at FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        if not row:
            return None, "account_not_found"
        if not self._verify_password(password, row["password_hash"]):
            return None, "password_mismatch"
        return (
            {
                "id": row["id"],
                "email": row["email"],
                "plan": row["plan"],
                "stripe_customer_id": row["stripe_customer_id"],
                "stripe_subscription_id": row["stripe_subscription_id"],
                "created_at": row["created_at"],
            },
            "ok",
        )

    def authenticate_user(self, email: str, password: str) -> dict | None:
        user, _reason = self.authenticate_user_detailed(email=email, password=password)
        return user

    def change_password(self, user_id: int, current_password: str, new_password: str) -> bool:
        if len(new_password or "") < 8:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT password_hash FROM users WHERE id = ?",
                (user_id,),
            ).fetchone()
            if not row:
                return False
            if not self._verify_password(current_password, row["password_hash"]):
                return False
            new_hash = self._hash_password(new_password)
            cur = conn.execute(
                "UPDATE users SET password_hash = ? WHERE id = ?",
                (new_hash, user_id),
            )
            return cur.rowcount > 0

    def create_password_reset_token(self, email: str, ttl_minutes: int = 30) -> str | None:
        normalized = (email or "").strip().lower()
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
            if not row:
                return None
            user_id = int(row["id"])
            token = secrets.token_urlsafe(40)
            expires_at = datetime.now(timezone.utc) + timedelta(minutes=max(ttl_minutes, 5))
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute(
                "INSERT INTO password_reset_tokens (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at.strftime("%Y-%m-%d %H:%M:%S")),
            )
            return token

    def reset_password_with_token(self, token: str, new_password: str) -> bool:
        if len(new_password or "") < 8:
            return False
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT user_id
                FROM password_reset_tokens
                WHERE token = ? AND expires_at > ?
                """,
                ((token or "").strip(), now),
            ).fetchone()
            if not row:
                return False
            user_id = int(row["user_id"])
            new_hash = self._hash_password(new_password)
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
            return True

    def reset_password_by_email(self, email: str, new_password: str) -> bool:
        normalized = (email or "").strip().lower()
        if not normalized or len(new_password or "") < 8:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
            if not row:
                return False
            user_id = int(row["id"])
            new_hash = self._hash_password(new_password)
            conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (new_hash, user_id))
            conn.execute("DELETE FROM password_reset_tokens WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
            return True

    def create_session(self, user_id: int, ttl_days: int = 180) -> str:
        token = secrets.token_urlsafe(48)
        expires_at = datetime.now(timezone.utc) + timedelta(days=max(ttl_days, 1))
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO user_sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
                (token, user_id, expires_at.strftime("%Y-%m-%d %H:%M:%S")),
            )
        return token

    def get_user_by_token(self, token: str, max_age_days: int | None = None) -> dict | None:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        age_clause = ""
        params: list[object] = [token, now]
        if max_age_days is not None:
            age_clause = " AND s.created_at >= datetime('now', ?)"
            params.append(f"-{max(max_age_days, 1)} days")
        with self._connect() as conn:
            row = conn.execute(
                f"""
                SELECT u.id, u.email, u.plan, u.created_at, s.expires_at
                       ,u.stripe_customer_id, u.stripe_subscription_id, s.created_at AS session_created_at
                FROM user_sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ? AND s.expires_at > ?
                {age_clause}
                """,
                tuple(params),
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

    def delete_sessions_for_user(self, user_id: int) -> int:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
            return int(cur.rowcount or 0)

    def touch_session(self, token: str, ttl_days: int = 180) -> bool:
        new_expiry = datetime.now(timezone.utc) + timedelta(days=max(ttl_days, 1))
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE user_sessions
                SET expires_at = ?
                WHERE token = ? AND expires_at > datetime('now')
                """,
                (new_expiry.strftime("%Y-%m-%d %H:%M:%S"), token),
            )
            return cur.rowcount > 0

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

    def admin_stats(self) -> dict:
        with self._connect() as conn:
            users_total = conn.execute("SELECT COUNT(1) AS cnt FROM users").fetchone()["cnt"]
            conversations_total = conn.execute("SELECT COUNT(1) AS cnt FROM conversations").fetchone()["cnt"]
            messages_total = conn.execute("SELECT COUNT(1) AS cnt FROM messages").fetchone()["cnt"]
            active_last_24h = conn.execute(
                """
                SELECT COUNT(DISTINCT user_id) AS cnt
                FROM usage_events
                WHERE created_at >= datetime('now', '-24 hour')
                """
            ).fetchone()["cnt"]
            plan_rows = conn.execute(
                "SELECT plan, COUNT(1) AS cnt FROM users GROUP BY plan ORDER BY plan ASC"
            ).fetchall()
            plans = {row["plan"]: int(row["cnt"]) for row in plan_rows}
        return {
            "users_total": int(users_total or 0),
            "conversations_total": int(conversations_total or 0),
            "messages_total": int(messages_total or 0),
            "active_users_last_24h": int(active_last_24h or 0),
            "plans": plans,
        }

    def set_conversation_title(self, conversation_id: str, title: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                ((title or "")[:80], conversation_id),
            )

    def delete_conversation(self, conversation_id: str, user_id: int | None = None) -> bool:
        with self._connect() as conn:
            if user_id is not None:
                row = conn.execute(
                    "SELECT user_id FROM conversations WHERE id = ?",
                    (conversation_id,),
                ).fetchone()
                if not row or row["user_id"] != user_id:
                    return False
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM user_profiles WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM uploaded_files WHERE conversation_id = ?", (conversation_id,))
            result = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
            return result.rowcount > 0

    def add_user_knowledge_file(
        self,
        user_id: int,
        filename: str,
        content: str,
        workspace_id: str | None = None,
    ) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO user_knowledge_files (user_id, filename, content, workspace_id) VALUES (?, ?, ?, ?)",
                (user_id, filename, content, workspace_id),
            )
            return int(cursor.lastrowid)

    def list_user_knowledge_files(self, user_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, created_at FROM user_knowledge_files WHERE user_id = ? ORDER BY id DESC LIMIT 20",
                (user_id,),
            ).fetchall()
        return [{"id": row["id"], "filename": row["filename"], "created_at": row["created_at"]} for row in rows]

    def get_user_knowledge_content(self, user_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, filename, content FROM user_knowledge_files WHERE user_id = ? ORDER BY id DESC LIMIT 10",
                (user_id,),
            ).fetchall()
        return [{"id": row["id"], "filename": row["filename"], "content": row["content"]} for row in rows]

    def delete_user_knowledge_file(self, file_id: int, user_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM user_knowledge_files WHERE id = ? AND user_id = ?",
                (file_id, user_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Workspaces
    # ------------------------------------------------------------------

    def get_user_by_email(self, email: str) -> dict | None:
        normalized = (email or "").strip().lower()
        if not normalized:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, email, plan, created_at FROM users WHERE email = ?",
                (normalized,),
            ).fetchone()
        return dict(row) if row else None

    def create_workspace(self, name: str, owner_user_id: int) -> dict:
        workspace_id = str(uuid.uuid4())
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO workspaces (id, name, owner_user_id) VALUES (?, ?, ?)",
                (workspace_id, (name or "").strip(), owner_user_id),
            )
            conn.execute(
                "INSERT INTO workspace_members (workspace_id, user_id, role) VALUES (?, ?, 'owner')",
                (workspace_id, owner_user_id),
            )
            row = conn.execute(
                "SELECT id, name, owner_user_id, created_at FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        return dict(row)

    def get_workspace(self, workspace_id: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, name, owner_user_id, created_at FROM workspaces WHERE id = ?",
                (workspace_id,),
            ).fetchone()
        return dict(row) if row else None

    def list_user_workspaces(self, user_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT w.id, w.name, w.owner_user_id, w.created_at, wm.role
                FROM workspaces w
                JOIN workspace_members wm ON wm.workspace_id = w.id
                WHERE wm.user_id = ?
                ORDER BY w.created_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_workspace_member(self, workspace_id: str, user_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT workspace_id, user_id, role, joined_at FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            ).fetchone()
        return dict(row) if row else None

    def add_workspace_member(self, workspace_id: str, user_id: int, role: str = "member") -> bool:
        """Add a member; returns False if already a member."""
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT 1 FROM workspace_members WHERE workspace_id = ? AND user_id = ?",
                (workspace_id, user_id),
            ).fetchone()
            if existing:
                return False
            conn.execute(
                "INSERT INTO workspace_members (workspace_id, user_id, role) VALUES (?, ?, ?)",
                (workspace_id, user_id, role),
            )
        return True

    def list_workspace_knowledge_files(self, workspace_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT f.id, f.filename, f.user_id, f.created_at
                FROM user_knowledge_files f
                WHERE f.workspace_id = ?
                ORDER BY f.id DESC
                LIMIT 50
                """,
                (workspace_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_workspace_knowledge_content(self, workspace_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, filename, content
                FROM user_knowledge_files
                WHERE workspace_id = ?
                ORDER BY id DESC
                LIMIT 10
                """,
                (workspace_id,),
            ).fetchall()
        return [{"id": row["id"], "filename": row["filename"], "content": row["content"]} for row in rows]

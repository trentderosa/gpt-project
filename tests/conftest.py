"""Shared pytest fixtures for Cortex Engine.

CRITICAL: this file sets environment variables *before* importing gpt_project.api,
because gpt_project.core.config / storage / api all read os.environ at *module import
time* to build module-level singletons (DB path, LLM client, search provider, etc).
Getting the import order wrong means tests would run against production defaults.

Tests never touch the real gpt_project.db — DB_PATH is forced to a fresh temp file
for the whole test session.
"""
import os
import tempfile
import uuid
from pathlib import Path

import pytest

_TEST_DB_DIR = Path(tempfile.mkdtemp(prefix="cortex_test_db_"))
os.environ["DB_PATH"] = str(_TEST_DB_DIR / "test_cortex.db")
os.environ["OPENAI_API_KEY"] = "sk-test-dummy-not-a-real-key"
os.environ["LIVE_UPDATE_ENABLED"] = "false"
os.environ["RUN_UPDATER_IN_API"] = "false"
os.environ["WEB_SEARCH_PROVIDER"] = "disabled"
os.environ["PASSWORD_RESET_DEV_MODE"] = "true"
os.environ["ALLOWED_ORIGINS"] = ""
os.environ.pop("CREATOR_EMAIL", None)
os.environ.pop("CREATOR_PASSWORD", None)
os.environ.pop("CREATOR_BOOTSTRAP_SECRET", None)
os.environ.pop("STRIPE_SECRET_KEY", None)
os.environ.pop("STRIPE_WEBHOOK_SECRET", None)

from starlette.testclient import TestClient  # noqa: E402

from gpt_project.api import app  # noqa: E402


@pytest.fixture(scope="session")
def api_client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def client(api_client):
    return api_client


def unique_email(prefix: str = "user") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}@example.com"


DEFAULT_TEST_PASSWORD = "correct horse battery staple 1"


@pytest.fixture()
def make_user(client):
    """Factory fixture: register a fresh user and return their auth context."""

    def _make(prefix: str = "user") -> dict:
        email = unique_email(prefix)
        resp = client.post(
            "/auth/register",
            json={"email": email, "password": DEFAULT_TEST_PASSWORD},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        return {
            "email": email,
            "password": DEFAULT_TEST_PASSWORD,
            "token": body["token"],
            "user": body["user"],
            "headers": {"Authorization": f"Bearer {body['token']}"},
        }

    return _make

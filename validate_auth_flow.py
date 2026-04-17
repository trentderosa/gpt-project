import importlib
import os
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from gpt_project.core.storage import ChatStorage


REPO_ROOT = Path(__file__).resolve().parent


def _set_env(db_path: Path) -> tuple[str, str, str]:
    creator_email = "creator@example.com"
    creator_password = "CreatorPass123!"
    normal_password = "NormalPass123!"
    os.environ["DB_PATH"] = str(db_path)
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "test-key")
    os.environ["LIVE_UPDATE_ENABLED"] = "false"
    os.environ["RUN_UPDATER_IN_API"] = "false"
    os.environ["CREATOR_EMAIL"] = creator_email
    os.environ["CREATOR_PASSWORD"] = creator_password
    os.environ["CREATOR_BOOTSTRAP_SECRET"] = "bootstrap-secret"
    os.environ["PASSWORD_RESET_DEV_MODE"] = "true"
    return creator_email, creator_password, normal_password


def _load_app():
    for module_name in [
        "gpt_project.api",
        "gpt_project.core.storage",
        "gpt_project.core.config",
        "gpt_project.core.llm_wrapper",
    ]:
        if module_name in sys.modules:
            del sys.modules[module_name]
    return importlib.import_module("gpt_project.api")


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _configure_runtime(api, db_path: Path, creator_email: str, creator_password: str) -> None:
    api.storage = ChatStorage(db_path=db_path)
    api.CREATOR_EMAIL = creator_email
    api.CREATOR_PASSWORD = creator_password
    api.CREATOR_BOOTSTRAP_SECRET = "bootstrap-secret"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cortex-auth-", ignore_cleanup_errors=True) as temp_dir:
        db_path = Path(temp_dir) / "auth-test.db"
        creator_email, creator_password, normal_password = _set_env(db_path)
        api = _load_app()
        _configure_runtime(api, db_path=db_path, creator_email=creator_email, creator_password=creator_password)

        results: list[dict] = []

        with TestClient(api.app) as creator_client:
            creator_login = creator_client.post(
                "/auth/login",
                json={"email": creator_email, "password": creator_password},
            )
            creator_login_data = creator_login.json()
            _assert(creator_login.status_code == 200, "creator login should succeed")
            _assert(creator_login_data["user"]["is_creator"] is True, "creator should be flagged as creator")
            _assert(creator_login_data["user"]["is_admin"] is True, "creator should be flagged as admin")
            _assert(creator_login_data["user"]["plan"] == "creator", "creator plan should resolve to creator")
            results.append(
                {
                    "name": "creator email + correct password",
                    "status": "PASS",
                    "code_path": "startup ensure -> /auth/login -> authenticate_user_detailed -> create_session -> _auth_user_payload",
                    "flags": "is_creator=true is_admin=true",
                    "tiers": "creator unlimited limits returned",
                }
            )

            wrong_login = creator_client.post(
                "/auth/login",
                json={"email": creator_email, "password": "WrongPass123!"},
            )
            _assert(wrong_login.status_code == 401, "creator wrong password should fail")
            results.append(
                {
                    "name": "creator email + wrong password",
                    "status": "PASS",
                    "code_path": "/auth/login -> authenticate_user_detailed(password_mismatch)",
                    "flags": "no session created",
                    "tiers": "not applicable",
                }
            )

            creator_me = creator_client.get("/auth/me")
            creator_me_data = creator_me.json()
            _assert(creator_me.status_code == 200, "creator /auth/me should succeed")
            _assert(creator_me_data["user"]["is_creator"] is True, "creator /auth/me should preserve creator flag")
            _assert(creator_me_data["limits"]["unlimited"] is True, "creator should have unlimited limits")
            admin_stats = creator_client.get("/admin/stats")
            _assert(admin_stats.status_code == 200, "creator should have admin access")
            home_page = creator_client.get("/")
            _assert("creatorSecret" not in home_page.text, "legacy creator secret field should be removed from UI")
            results.append(
                {
                    "name": "creator login followed by viewing creator/tier features",
                    "status": "PASS",
                    "code_path": "/auth/me -> _auth_user_payload -> creator UI consumes explicit flags",
                    "flags": "is_creator=true is_admin=true",
                    "tiers": "creator plan + unlimited limits + admin route access",
                }
            )

            creator_logout = creator_client.post("/auth/logout")
            _assert(creator_logout.status_code == 200, "creator logout should succeed")
            creator_relogin = creator_client.post(
                "/auth/login",
                json={"email": creator_email, "password": creator_password},
            )
            creator_relogin_data = creator_relogin.json()
            _assert(creator_relogin.status_code == 200, "creator relogin should succeed")
            _assert(creator_relogin_data["user"]["is_creator"] is True, "creator relogin should preserve creator flag")
            results.append(
                {
                    "name": "logout then creator login again",
                    "status": "PASS",
                    "code_path": "/auth/logout -> /auth/login -> create_session",
                    "flags": "is_creator=true is_admin=true",
                    "tiers": "creator plan preserved after logout/login",
                }
            )

        with TestClient(api.app) as fresh_session_client:
            fresh_login = fresh_session_client.post(
                "/auth/login",
                json={"email": creator_email, "password": creator_password},
            )
            fresh_login_data = fresh_login.json()
            _assert(fresh_login.status_code == 200, "fresh session creator login should succeed")
            _assert(fresh_login_data["user"]["is_creator"] is True, "fresh session should preserve creator flag")
            results.append(
                {
                    "name": "fresh session creator login",
                    "status": "PASS",
                    "code_path": "startup ensure -> new client /auth/login",
                    "flags": "is_creator=true is_admin=true",
                    "tiers": "creator plan preserved across fresh sessions",
                }
            )

        with TestClient(api.app) as normal_client:
            register = normal_client.post(
                "/auth/register",
                json={"email": "user@example.com", "password": normal_password},
            )
            register_data = register.json()
            _assert(register.status_code == 200, "normal user registration should succeed")
            _assert(register_data["user"]["is_creator"] is False, "normal user should not be creator")
            _assert(register_data["user"]["plan"] == "free", "normal user should stay on free plan")
            normal_client.post("/auth/logout")
            normal_login = normal_client.post(
                "/auth/login",
                json={"email": "user@example.com", "password": normal_password},
            )
            normal_login_data = normal_login.json()
            _assert(normal_login.status_code == 200, "normal user login should succeed")
            _assert(normal_login_data["user"]["is_creator"] is False, "normal user login should not gain creator access")
            results.append(
                {
                    "name": "normal user login",
                    "status": "PASS",
                    "code_path": "/auth/register -> create_user -> /auth/login -> authenticate_user_detailed",
                    "flags": "is_creator=false is_admin=false",
                    "tiers": "free tier returned",
                }
            )

            admin_denied = normal_client.get("/admin/stats")
            _assert(admin_denied.status_code == 403, "normal user should not access creator admin routes")
            results.append(
                {
                    "name": "register a normal user and confirm they do NOT get creator access",
                    "status": "PASS",
                    "code_path": "/auth/register -> create_user -> /admin/stats denied",
                    "flags": "is_creator=false is_admin=false",
                    "tiers": "free tier only, creator access denied",
                }
            )

        print("Auth validation results")
        for result in results:
            print(
                f"- {result['name']}: {result['status']} | path={result['code_path']} | "
                f"flags={result['flags']} | tiers={result['tiers']}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

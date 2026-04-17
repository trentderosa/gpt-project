import importlib
import io
import os
import re
import sys
import tempfile
from pathlib import Path

from fastapi.testclient import TestClient

from gpt_project.core.storage import ChatStorage


class FakeLLM:
    def chat(self, messages, temperature=0.2, max_tokens=2048, model=None):
        joined = "\n".join(str(msg.get("content", "")) for msg in messages)
        if "Generate a short 4-6 word title" in joined:
            return "Workspace Memory"
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = str(msg.get("content", ""))
                break
        lowered = user_content.lower()
        if "apollo dashboard" in lowered and "what did we decide" not in lowered:
            return "Decision recorded: launch the Apollo dashboard on May 1, keep the tone direct, and draft the rollout plan next."
        if "what did we decide" in lowered or "project plan" in lowered:
            match = re.search(r"Relevant workspace memory:\n- \[[^\]]+\] (.+)", user_content)
            if match:
                return f"Workspace memory says: {match.group(1)}"
            return "No workspace memory available."
        if "budget" in lowered or "brand voice" in lowered or "workspace files" in lowered:
            match = re.search(r"Relevant workspace files:\n- \[[^\]]+\] (.+)", user_content)
            if match:
                return f"Workspace files say: {match.group(1)}"
            return "No workspace file memory available."
        return "General workspace response."

    def chat_stream(self, messages, temperature=0.2, max_tokens=2048, model=None):
        yield self.chat(messages, temperature=temperature, max_tokens=max_tokens, model=model)

    def generate_image(self, prompt, size="1024x1024", quality="medium"):
        return "data:image/png;base64,ZmFrZQ=="

    def analyze_image(self, image_data_url, instruction=None):
        return "Image analyzed."


def _set_env(db_path: Path) -> None:
    os.environ["DB_PATH"] = str(db_path)
    os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "test-key")
    os.environ["LIVE_UPDATE_ENABLED"] = "false"
    os.environ["RUN_UPDATER_IN_API"] = "false"


def _load_api():
    for module_name in ["gpt_project.api", "gpt_project.core.storage"]:
        if module_name in sys.modules:
            del sys.modules[module_name]
    return importlib.import_module("gpt_project.api")


def _configure_api(api, db_path: Path) -> None:
    api.storage = ChatStorage(db_path=db_path)
    fake = FakeLLM()
    api.llm = fake
    api.chat_service.llm = fake


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _register(client: TestClient, email: str, password: str) -> None:
    response = client.post("/auth/register", json={"email": email, "password": password})
    _assert(response.status_code == 200, f"register failed for {email}: {response.text}")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="cortex-workspace-", ignore_cleanup_errors=True) as temp_dir:
        db_path = Path(temp_dir) / "workspace-test.db"
        _set_env(db_path)
        api = _load_api()
        _configure_api(api, db_path)

        results: list[dict] = []

        with TestClient(api.app) as client_a, TestClient(api.app) as client_b:
            _register(client_a, "user1@example.com", "WorkspacePass123!")
            workspace_a = client_a.post("/workspaces", json={"name": "Apollo"}).json()
            chat1 = client_a.post(
                "/chat",
                json={
                    "message": "We decided to launch the Apollo dashboard on May 1. Next step: draft the rollout plan. Please keep the tone direct.",
                    "workspace_id": workspace_a["id"],
                    "use_web_search": False,
                },
            )
            _assert(chat1.status_code == 200, f"chat1 failed: {chat1.text}")

            chat2 = client_a.post(
                "/chat",
                json={
                    "message": "What did we decide about the Apollo dashboard plan?",
                    "workspace_id": workspace_a["id"],
                    "use_web_search": False,
                },
            )
            answer2 = chat2.json()["answer"]
            _assert("May 1" in answer2 or "Apollo dashboard" in answer2, "workspace A should recall prior plan")
            results.append(
                {
                    "name": "Workspace A chat 2 recalls chat 1 plan",
                    "status": "PASS",
                    "behavior": answer2,
                }
            )

            upload = client_a.post(
                "/knowledge",
                files={"file": ("apollo_notes.txt", io.BytesIO(b"Budget approved at $50k. Brand voice should stay direct and concise."), "text/plain")},
                data={"workspace_id": workspace_a["id"]},
            )
            _assert(upload.status_code == 200, f"workspace file upload failed: {upload.text}")
            chat3 = client_a.post(
                "/chat",
                json={
                    "message": "What do the workspace files say about the budget and brand voice?",
                    "workspace_id": workspace_a["id"],
                    "use_web_search": False,
                },
            )
            answer3 = chat3.json()["answer"]
            _assert("$50k" in answer3 or "Brand voice" in answer3 or "direct" in answer3, "workspace files should be reusable across chats")
            results.append(
                {
                    "name": "Workspace A file memory is reusable in chat 3",
                    "status": "PASS",
                    "behavior": answer3,
                }
            )

            workspace_b = client_a.post("/workspaces", json={"name": "Hermes"}).json()
            chat_b = client_a.post(
                "/chat",
                json={
                    "message": "What did we decide about the Apollo dashboard plan?",
                    "workspace_id": workspace_b["id"],
                    "use_web_search": False,
                },
            )
            answer_b = chat_b.json()["answer"]
            _assert("No workspace memory" in answer_b, "workspace B should not inherit workspace A context")
            results.append(
                {
                    "name": "Workspace B stays isolated from workspace A",
                    "status": "PASS",
                    "behavior": answer_b,
                }
            )

            _register(client_b, "user2@example.com", "WorkspacePass456!")
            workspace_c = client_b.post("/workspaces", json={"name": "Apollo"}).json()
            chat_c = client_b.post(
                "/chat",
                json={
                    "message": "What did we decide about the Apollo dashboard plan?",
                    "workspace_id": workspace_c["id"],
                    "use_web_search": False,
                },
            )
            answer_c = chat_c.json()["answer"]
            _assert("No workspace memory" in answer_c, "different user must not see workspace A memory")
            results.append(
                {
                    "name": "Different user workspace stays isolated",
                    "status": "PASS",
                    "behavior": answer_c,
                }
            )

            forbidden_detail = client_b.get(f"/workspaces/{workspace_a['id']}")
            _assert(forbidden_detail.status_code == 403, "different user should not access another workspace detail")
            forbidden_conversations = client_b.get(f"/conversations?workspace_id={workspace_a['id']}")
            _assert(forbidden_conversations.status_code == 403, "different user should not list another workspace conversations")
            results.append(
                {
                    "name": "Workspace API isolation blocks other users",
                    "status": "PASS",
                    "behavior": "Cross-user workspace detail and conversation access returned 403.",
                }
            )

            detail = client_a.get(f"/workspaces/{workspace_a['id']}")
            _assert(detail.status_code == 200, f"workspace detail failed: {detail.text}")
            detail_data = detail.json()
            _assert(detail_data.get("summary", {}).get("summary_text"), "workspace summary should be populated")
            _assert(len(detail_data.get("memory_items", [])) > 0, "workspace memory items should exist")
            results.append(
                {
                    "name": "Workspace detail exposes summary and memory items",
                    "status": "PASS",
                    "behavior": detail_data.get("summary", {}).get("summary_text", ""),
                }
            )

        print("Workspace memory validation results")
        for result in results:
            print(f"- {result['name']}: {result['status']} | {result['behavior']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

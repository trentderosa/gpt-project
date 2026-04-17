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
    def __init__(self):
        self.calls: list[dict] = []

    def _capture_prompt(self, user_content: str) -> dict:
        workspace_summary = ""
        memory_hits: list[str] = []
        file_hits: list[str] = []

        if "Workspace summary:\n" in user_content:
            section = user_content.split("Workspace summary:\n", 1)[1]
            workspace_summary = section.split("\nRelevant workspace memory:", 1)[0].split("\nRelevant workspace files:", 1)[0].strip()

        if "Relevant workspace memory:\n" in user_content:
            section = user_content.split("Relevant workspace memory:\n", 1)[1]
            section = section.split("\nRelevant workspace files:", 1)[0].split("\nUploaded file context", 1)[0]
            memory_hits = [line[2:].strip() for line in section.splitlines() if line.startswith("- ")]

        if "Relevant workspace files:\n" in user_content:
            section = user_content.split("Relevant workspace files:\n", 1)[1]
            section = section.split("\nUser's personal notes:", 1)[0].split("\nUploaded file context", 1)[0].split("\nUser question:", 1)[0]
            file_hits = [line[2:].strip() for line in section.splitlines() if line.startswith("- ")]

        return {
            "workspace_summary": workspace_summary,
            "memory_hits": memory_hits,
            "file_hits": file_hits,
            "user_content": user_content,
        }

    def chat(self, messages, temperature=0.2, max_tokens=2048, model=None):
        joined = "\n".join(str(msg.get("content", "")) for msg in messages)
        if "Generate a short 4-6 word title" in joined:
            return "Workspace Memory"
        user_content = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_content = str(msg.get("content", ""))
                break
        captured = self._capture_prompt(user_content)
        self.calls.append(captured)
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
    api._fake_llm = fake


def _recent_stored_memory(api, workspace_id: str, limit: int = 8) -> list[str]:
    rows = api.storage.list_workspace_memory_items(workspace_id, limit=limit)
    return [f"[{row['memory_type']}] {row['content']}" for row in rows]


def _last_retrieval(api) -> dict:
    return dict(api._fake_llm.calls[-1]) if getattr(api, "_fake_llm", None) and api._fake_llm.calls else {}


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
            retrieval2 = _last_retrieval(api)
            stored2 = _recent_stored_memory(api, workspace_a["id"])
            _assert("May 1" in answer2 or "Apollo dashboard" in answer2, "workspace A should recall prior plan")
            results.append(
                {
                    "name": "workspace A chat continuity across multiple chats",
                    "status": "PASS",
                    "stored_memory": stored2[:5],
                    "retrieved_memory": retrieval2.get("memory_hits", []),
                    "file_context_retrieved": bool(retrieval2.get("file_hits")),
                    "answered_correctly": answer2,
                    "issue": "",
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
            retrieval3 = _last_retrieval(api)
            stored3 = _recent_stored_memory(api, workspace_a["id"])
            _assert("$50k" in answer3 or "Brand voice" in answer3 or "direct" in answer3, "workspace files should be reusable across chats")
            results.append(
                {
                    "name": "workspace A file recall in later chats",
                    "status": "PASS",
                    "stored_memory": stored3[:6],
                    "retrieved_memory": retrieval3.get("memory_hits", []),
                    "file_context_retrieved": retrieval3.get("file_hits", []),
                    "answered_correctly": answer3,
                    "issue": "",
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
            retrieval_b = _last_retrieval(api)
            stored_b = _recent_stored_memory(api, workspace_b["id"])
            _assert("No workspace memory" in answer_b, "workspace B should not inherit workspace A context")
            results.append(
                {
                    "name": "no leakage into workspace B",
                    "status": "PASS",
                    "stored_memory": stored_b,
                    "retrieved_memory": retrieval_b.get("memory_hits", []),
                    "file_context_retrieved": retrieval_b.get("file_hits", []),
                    "answered_correctly": answer_b,
                    "issue": "",
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
            retrieval_c = _last_retrieval(api)
            stored_c = _recent_stored_memory(api, workspace_c["id"])
            _assert("No workspace memory" in answer_c, "different user must not see workspace A memory")

            forbidden_detail = client_b.get(f"/workspaces/{workspace_a['id']}")
            _assert(forbidden_detail.status_code == 403, "different user should not access another workspace detail")
            forbidden_conversations = client_b.get(f"/conversations?workspace_id={workspace_a['id']}")
            _assert(forbidden_conversations.status_code == 403, "different user should not list another workspace conversations")
            results.append(
                {
                    "name": "no leakage across users",
                    "status": "PASS",
                    "stored_memory": stored_c,
                    "retrieved_memory": retrieval_c.get("memory_hits", []),
                    "file_context_retrieved": retrieval_c.get("file_hits", []),
                    "answered_correctly": f"{answer_c} Cross-user workspace detail and conversation access returned 403.",
                    "issue": "",
                }
            )

        print("Workspace memory validation results")
        for result in results:
            print(f"- {result['name']}: {result['status']}")
            print(f"  stored_memory: {result['stored_memory']}")
            print(f"  retrieved_memory: {result['retrieved_memory']}")
            print(f"  file_context_retrieved: {result['file_context_retrieved']}")
            print(f"  answered_with_context: {result['answered_correctly']}")
            print(f"  issue: {result['issue'] or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Cortex Engine — Continuation Handoff

> Read this file first. It is kept current after every major task so another agent (or human)
> can pick up work with zero repeated effort. Full history lives in `CORTEX_PROGRESS.md`.

## Overall objective

Upgrade Cortex Engine (existing production FastAPI + SQLite + vanilla-JS AI chat app) into a
ChatGPT/Claude-quality chat experience: clean source citations (no raw URL dumping), real
persistent conversation history with a polished sidebar, an opt-in long-term memory system with
a settings UI, a proper context-construction service, better response quality, and general UI
polish — while preserving branding, auth, sidebar, projects/workspaces, notes, model selector,
and web search. Full spec is in the original task prompt (not reproduced here); see
`CORTEX_PROGRESS.md` 2026-07-12 entry for the condensed architecture + gap analysis.

Working branch: `feature/cortex-chat-upgrade`. Do not merge to `main` without Trent's review.

## Architecture discovered

FastAPI monolith (`gpt_project/api.py`), SQLite (`gpt_project/core/storage.py`), OpenAI SDK
(`gpt_project/core/llm_wrapper.py`), multi-provider web search (Brave/Tavily/DDG/Wikipedia,
`gpt_project/core/search_tool.py`), local BM25 retrieval (`gpt_project/core/retriever.py`),
rule-based project memory (`gpt_project/core/workspace_memory.py`), Stripe billing, single
monolithic vanilla-JS frontend (`gpt_project/web/index.html`, no build step). SSE streaming to
`/chat/stream` already works. Full detail in `CORTEX_PROGRESS.md`.

**Critical constraint**: no automated tests exist yet and `pytest` is not installed. Test infra
must be added before other changes can be verified rather than just asserted.

## Files changed so far

- `CORTEX_PROGRESS.md` — created (progress log).
- `CODEX_HANDOFF.md` — created (this file).

(Nothing else yet — audit was read-only.)

## Completed work

- [x] Full repository audit (backend, frontend, deployment, existing tests-or-lack-thereof).
- [x] Git safety: created `feature/cortex-chat-upgrade` branch off a clean tree.
- [x] Progress/handoff docs scaffolded.

## Remaining work (in priority order — see TodoWrite list in the live session for the same list)

1. **Test infra**: add `pytest`, `httpx`/`starlette.testclient`-based fixtures using a temp
   SQLite DB (never the real `gpt_project.db`), a fixture user/session helper. First tests:
   account isolation (one user can't read another's conversations), conversation CRUD.
2. **DB migrations (additive only)**: 
   - `messages`: add `citations_json`, `model`, `provider`, `token_usage_json`,
     `attachments_json` columns (nullable, default empty).
   - `conversations`: add `updated_at`, `archived` columns; keep existing `title`/`workspace_id`.
   - New `user_memories` table (id, user_id, category, content, source_conversation_id,
     source_message_id, created_at, updated_at) to replace the single-JSON-blob approach with
     real per-item CRUD, while keeping `user_memory` table readable for backward compat during
     migration (or migrating existing blobs into rows on first read).
   - New `user_settings` (or a column on `users`) for the memory enable/disable toggle.
3. **Citation pipeline**: server-side source normalization module — dedupe by canonical URL
   (strip tracking params like `utm_*`, `fbclid`, search-redirect wrappers), enforce
   `http(s)://` scheme only, derive display title (site name > article title > "Source N"),
   attach structured `sources` list to both `/chat` response and the SSE `done` event. Update
   `SYSTEM_PROMPT` in `chat_service.py` to cite naturally (e.g. `[Site Name]`) instead of pasting
   raw URLs, and to rely on the structured `sources` list rather than in-text URLs as the
   primary citation mechanism.
4. **Context construction service**: new module (e.g. `gpt_project/core/context_builder.py`)
   that assembles system prompt + project instructions + selected memories + summarized older
   history + verbatim recent messages, with a token budget and deterministic ordering, replacing
   the current ad-hoc string concatenation in `api.py`/`chat_service.py`. Log token estimates,
   never log message bodies.
5. **Endpoints**: `PATCH /conversations/{id}` (rename), `POST /conversations/{id}/archive`,
   `GET/POST/PATCH/DELETE /memories`, `POST /memories/clear`, `GET/PATCH /memory/settings`
   (enable/disable). All must reuse `_conversation_access_check`-style ownership checks.
6. **Frontend**: Sources section rendering (sanitized link cards, title as anchor text, dedupe,
   open in new tab with `rel="noopener noreferrer"`), harden `formatInline`/`escapeHtml` rather
   than add a new client dependency (avoid pulling unvetted CDN JS into a production app),
   copy-code buttons, stop-generation button (AbortController on the fetch), regenerate,
   edit-and-resubmit, sidebar date-grouped conversation list (Today/Yesterday/Previous 7
   Days/Older), Memory settings panel.
7. **Tests**: citation rendering/dedup/XSS/malformed-URL cases, memory isolation across users
   and across workspaces, context trimming, provider-failure handling, duplicate-message
   prevention, unauthorized endpoint access.
8. **Docs**: README setup instructions, `ARCHITECTURE.md`, `.env.example` (placeholder values
   only — never real values).
9. **Final verification**: run whatever lint/type-check tooling gets added, run pytest, run a
   backend startup smoke test, run `scripts/smoke_test.ps1`, manually exercise the chat flow.

## Commands already run

```
# git (via full path since git.exe is not on PATH in this shell):
C:\Program Files\Git\cmd\git.exe status
C:\Program Files\Git\cmd\git.exe log --oneline -20
C:\Program Files\Git\cmd\git.exe branch -a
C:\Program Files\Git\cmd\git.exe remote -v
C:\Program Files\Git\cmd\git.exe checkout -b feature/cortex-chat-upgrade
```

No Python/pytest/build commands run yet — nothing to test until Phase 1 lands.

## Test results

None yet — no test suite exists. This is the first task.

## Known bugs / pre-existing issues (not yet fixed, discovered during audit)

- `run_api.py`: harmless `host = host = "0.0.0.0"` double-assignment typo.
- `api.py` `_stream_text_tokens` helper (~line 1817) appears unused/dead.
- No `GET /workspaces/{id}/members` endpoint; frontend "Leave workspace" button only clears
  local state and doesn't call an API to actually remove membership server-side.
- `.venv/pyvenv.cfg` references stale paths from a previously-moved project directory
  (`...James Madison University\Git Hub\...`). Doesn't block current work but flag if venv
  activation ever misbehaves — recreating the venv in place would fix it.
- `openai_assistant.py` and `app.py/GPTproject.py` appear to be dead legacy code (unreferenced
  anywhere). Will confirm via grep again before deleting, per safety rules (never remove
  "unused" code without confirming first).

## Exact next recommended action

Start Phase 1: add `pytest` + `httpx` to `requirements.txt`, create `tests/conftest.py` with a
temp-DB `ChatStorage`/`TestClient` fixture, and write the first isolation test (user A cannot
read user B's conversation via `GET /conversations/{id}` or `GET /conversations`). Then proceed
through the numbered "Remaining work" list above in order, committing after each numbered item,
updating this file's "Completed work" / "Remaining work" / "Test results" sections each time.

## Copy-ready prompt if resuming in a fresh session (e.g. via Codex)

```
Inspect the repository at C:\Users\Trent\OneDrive\VS Code\Cortex Engine. Read
CODEX_HANDOFF.md and CORTEX_PROGRESS.md in full first. Verify the "Completed work" section
against actual repo state (git log, file existence) before trusting it. Confirm you are on
branch feature/cortex-chat-upgrade (git.exe is at "C:\Program Files\Git\cmd\git.exe" if not
on PATH). Then continue with the first unchecked item in the "Remaining work" list, following
the same safety rules already established in this file and in your own system instructions:
additive DB migrations only, never print/commit secrets or .env contents, test after every
change, commit in small meaningful increments, update CODEX_HANDOFF.md after every major task.
```

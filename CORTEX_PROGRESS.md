# Cortex Engine — Chat Upgrade Progress Log

Branch: `feature/cortex-chat-upgrade` (created from `agent/update-deployment-config` @ `700e746`, which is 1 commit ahead of `main`).

This log is append-only, newest entry at the top. See `CODEX_HANDOFF.md` for the always-current
"pick up here" summary — this file is the historical record.

---

## 2026-07-12 — Session 1: Audit + plan

### Safety setup
- Working tree was already clean at session start (no baseline commit needed).
- Created branch `feature/cortex-chat-upgrade` off the current tip of
  `agent/update-deployment-config` (`700e746`), which is itself 1 commit ahead of `main`
  ("Update deployment configuration"). This is the most current, deployed-equivalent state.
- **Secret-safety note**: during the repo audit, a research subagent ran a broad grep against
  `.env` in its own isolated tool call and briefly saw the raw values for `OPENAI_API_KEY`,
  `CREATOR_PASSWORD`, and `CREATOR_BOOTSTRAP_SECRET` inside its own transcript. It did **not**
  relay those values back into this conversation (only variable *names* were reported), so no
  secret values entered this session's context, were printed to the user, or were committed.
  Out of an abundance of caution, **you (Trent) should rotate `OPENAI_API_KEY`,
  `CREATOR_PASSWORD`, and `CREATOR_BOOTSTRAP_SECRET`** the next time you touch Render env vars.
  This is a precaution, not a confirmed leak.

### Architecture discovered (see ARCHITECTURE.md once written, and CODEX_HANDOFF.md for the summary)
- **Backend**: FastAPI (`gpt_project/api.py`, ~2100 lines, monolithic router), SQLite
  (`gpt_project/core/storage.py`), OpenAI SDK (`gpt_project/core/llm_wrapper.py`), multi-provider
  web search (Brave/Tavily/DuckDuckGo/Wikipedia composite, `gpt_project/core/search_tool.py`),
  local BM25 knowledge retrieval (`gpt_project/core/retriever.py`), rule-based workspace/project
  memory (`gpt_project/core/workspace_memory.py`), Stripe billing, APScheduler-driven live
  data updater (stocks/news) embedded in the API process.
- **Frontend**: single monolithic `gpt_project/web/index.html` (~2300 lines: ~800 lines CSS,
  ~1360 lines vanilla JS). No build step, no framework, no npm. Hand-rolled markdown-lite
  renderer with HTML-escape-then-whitelist approach (no DOMPurify/marked.js). SSE streaming to
  `/chat/stream` already implemented (token-by-token) via manual `fetch` + `ReadableStream`
  reader (not native `EventSource`, since it needs a POST body).
- **Auth**: email/password, PBKDF2-HMAC-SHA256 hashing, opaque session tokens in httponly
  cookies + bearer-token support, sliding-window rate limiters, creator-account bootstrap.
- **Conversations/messages**: already persisted in SQLite (`conversations`, `messages` tables),
  already scoped to `user_id` with ownership checks (`_conversation_access_check`), anonymous
  pre-login conversations supported and claimable after login. **Gaps**: no `updated_at`,
  `archived`/`deleted` columns on conversations; no rename endpoint; messages table has no
  columns for citations, model/provider metadata, attachment metadata, or token usage.
- **Memory**: two systems already exist —
  1. **User memory** (`user_memory` table): a single JSON blob per user with `name`, `facts[]`,
     inferred `style`. Whole-profile only; `/memory` (GET) and `/memory/forget` (POST, delete by
     key) are the only endpoints. No categories, no per-item CRUD, no enable/disable toggle, no
     source conversation ID/timestamp traceability, no settings UI.
  2. **Workspace/project memory** (`workspace_memory_items`, `workspace_memory_summaries`
     tables): regex/keyword-classified items (`task`/`decision`/`preference`/`fact`/`file`/
     `summary`) scoped per workspace, BM25-ranked retrieval, rolling summary. This is a solid,
     already-isolated implementation of Feature 3's "project-specific context" layer (D).
     Documented in `WORKSPACE_MEMORY.md`.
- **Citations today**: the backend returns `retrieved_sources` (local KB hits) and `web_results`
  (web search hits: title/url/snippet/date) on both `/chat` and the `done` SSE event of
  `/chat/stream` — but **the frontend never renders either field**. The system prompt instructs
  the model to "cite the most relevant source URLs inline," so raw URLs end up pasted directly
  into assistant answer text, which the frontend's `formatInline` linkifier then turns into
  plain `<a>` tags. There is no dedicated "Sources" section, no dedup, no tracking-param
  stripping, no title-as-anchor-text.
- **No automated test suite exists at all.** `pytest` isn't installed or in `requirements.txt`.
  Only manual validation scripts (`validate_auth_flow.py`, `validate_pipeline.py`,
  `validate_workspace_memory.py`, `scripts/smoke_test.ps1`).
- **Dead/legacy code identified** (not yet removed — will confirm-then-remove per safety rules):
  `openai_assistant.py` (root, unused standalone script), `app.py/GPTproject.py` (unused
  compatibility launcher; note `app.py/knowledge/` itself IS live — it's the BM25 knowledge dir).
  Also `_stream_text_tokens` helper in `api.py` (~line 1817) appears unused.
- **Deployment**: Render, single web service, persistent disk SQLite at `/var/data`, live
  updater runs embedded in the API process (`RUN_UPDATER_IN_API=true`). `run_worker.py` is an
  alternate standalone-worker entrypoint not used by the current Render config.

### Plan (see CODEX_HANDOFF.md "Remaining work" for live status)
1. Test infra first (pytest + fixtures against a temp SQLite DB) so every later change is
   actually verifiable, per the "never claim something works without testing it" rule.
2. Additive DB migrations (new columns/tables only, no destructive changes) for citations,
   message metadata, conversation archive/rename, categorized+per-item user memory.
3. Backend citation pipeline: normalize/dedupe/sanitize source metadata server-side, stop
   relying on the model to format links correctly, update system prompt to cite naturally
   instead of pasting raw URLs.
4. Backend context-construction service + response-quality system-prompt updates.
5. Backend endpoints: conversation rename, memory per-item CRUD + categories + enable/disable.
6. Frontend: render a proper Sources section (sanitized, deduped, safe-protocol-only links),
   harden the existing hand-rolled sanitizer rather than adding a new client dependency,
   copy-code buttons, stop-generation, regenerate/edit-resubmit, sidebar date grouping, a
   Memory settings panel.
7. Tests for citations/XSS/isolation/context-trimming/provider-failure per Feature 7.
8. Docs: README, ARCHITECTURE.md, `.env.example` (fake values only).
9. Full verification pass (lint/typecheck-equivalent/tests/build/smoke) + final report.

Status: **Phase 0 (audit) complete.** Beginning Phase 1 (test infra).

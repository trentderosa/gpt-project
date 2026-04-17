# Workspace Memory

## Overview

Workspaces now act like long-lived project contexts instead of temporary chat buckets.

Each workspace has:
- shared conversation memory across chats
- shared file memory across chats
- a rolling workspace summary
- isolated retrieval scoped to that workspace only

## Storage model

The workspace memory system uses:
- `conversations.workspace_id`
  - links a chat thread to a workspace
- `workspace_memory_items`
  - durable memory records extracted from prior chats and workspace files
  - includes `workspace_id`, `user_id`, `source_conversation_id`, `source_message_id`, `memory_type`, `title`, `content`, `keywords_json`, and `importance`
- `workspace_memory_summaries`
  - one rolling summary row per workspace
- `user_knowledge_files.workspace_id`
  - workspace-level note/file storage
- `uploaded_files`
  - still tied to conversations, but workspace uploads are reused by joining through workspace-linked conversations

## Memory ingestion

After each workspace chat turn:
1. the user and assistant messages are stored normally
2. durable workspace memory items are extracted
3. the workspace summary is refreshed

Workspace file uploads also create file-memory entries so future chats can use them.

## Retrieval

When a chat runs inside a workspace, the prompt includes:
1. current chat history
2. workspace summary
3. relevant workspace memory hits
4. relevant workspace files
5. current-chat uploaded files
6. personal user notes

Workspace retrieval uses lexical ranking over:
- `workspace_memory_items`
- workspace knowledge files
- uploaded files from conversations linked to the same workspace

## Isolation rules

- workspace memory is only visible to members of that workspace
- workspace A does not leak into workspace B
- other users do not inherit memory from someone else's workspace
- non-workspace chats do not import all workspace files anymore

## UX behavior

- selecting a workspace activates shared memory for new chats
- new chats created while a workspace is active inherit that workspace context
- workspace files are visible from the workspace panel
- the workspace summary is shown in the active workspace panel

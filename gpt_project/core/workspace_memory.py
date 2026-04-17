import re
from collections import OrderedDict


_TOKEN_RE = re.compile(r"[a-zA-Z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "how", "i", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "we", "what", "when", "where", "who", "why",
    "with", "you", "your", "our", "their",
}
_TASK_RE = re.compile(r"\b(?:todo|to do|action item|next step|need to|needs to|should|must|follow up|plan to|will need)\b", re.IGNORECASE)
_DECISION_RE = re.compile(r"\b(?:decided|decision|we will|we'll|going with|choose|chosen|selected|ship with|use the|final plan)\b", re.IGNORECASE)
_PREFERENCE_RE = re.compile(r"\b(?:prefer|preference|avoid|style|tone|format|naming|convention|voice)\b", re.IGNORECASE)
_GOAL_RE = re.compile(r"\b(?:goal|objective|milestone|target|deadline|launch|ship|roadmap|project plan)\b", re.IGNORECASE)


def _clean_text(text: str, limit: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip(" ,;:-") + "..."


def _sentences(text: str) -> list[str]:
    raw = _clean_text(text, limit=1000)
    if not raw:
        return []
    return [part.strip() for part in _SENTENCE_SPLIT_RE.split(raw) if part.strip()]


def summarize_text(text: str, max_chars: int = 240) -> str:
    parts = _sentences(text)
    if not parts:
        return ""
    summary = parts[0]
    if len(summary) < max_chars * 0.6 and len(parts) > 1:
        candidate = f"{summary} {parts[1]}"
        if len(candidate) <= max_chars:
            summary = candidate
    if len(summary) <= max_chars:
        return summary
    return summary[:max_chars].rstrip(" ,;:-") + "..."


def extract_keywords(text: str, limit: int = 12) -> list[str]:
    ordered: OrderedDict[str, None] = OrderedDict()
    for token in _TOKEN_RE.findall((text or "").lower()):
        if len(token) <= 2 or token in _STOPWORDS:
            continue
        ordered[token] = None
        if len(ordered) >= limit:
            break
    return list(ordered.keys())


def build_file_memory_item(filename: str, extracted_text: str) -> dict:
    summary = summarize_text(extracted_text, max_chars=260) or _clean_text(extracted_text, limit=260)
    content = f"{filename}: {summary}".strip()
    return {
        "memory_type": "file",
        "title": f"File: {filename}"[:80],
        "content": content[:600],
        "keywords": extract_keywords(f"{filename} {summary}"),
        "importance": 0.78,
    }


def extract_workspace_memory_items(
    question: str,
    answer: str,
    workspace_name: str | None = None,
) -> list[dict]:
    question_summary = summarize_text(question, max_chars=180)
    answer_summary = summarize_text(answer, max_chars=220)
    workspace_label = f"Workspace '{workspace_name}'" if workspace_name else "Workspace"
    items: list[dict] = []

    turn_summary = f"{workspace_label} turn summary: User asked about {question_summary}. Assistant response: {answer_summary}."
    items.append(
        {
            "memory_type": "summary",
            "title": "Conversation summary",
            "content": _clean_text(turn_summary, limit=620),
            "keywords": extract_keywords(f"{question_summary} {answer_summary}"),
            "importance": 0.52,
        }
    )

    combined_candidates = []
    for source, text in (("question", question), ("answer", answer)):
        for sentence in _sentences(text):
            combined_candidates.append((source, sentence))

    seen_contents: set[str] = set()
    for source, sentence in combined_candidates:
        lowered = sentence.lower()
        if len(sentence) < 18:
            continue
        memory_type = ""
        importance = 0.0
        if _DECISION_RE.search(sentence):
            memory_type = "decision"
            importance = 0.92
        elif _TASK_RE.search(sentence):
            memory_type = "task"
            importance = 0.88
        elif _PREFERENCE_RE.search(sentence):
            memory_type = "preference"
            importance = 0.82
        elif _GOAL_RE.search(sentence):
            memory_type = "fact"
            importance = 0.74
        if not memory_type:
            continue
        content = _clean_text(sentence, limit=420)
        dedupe_key = f"{memory_type}:{content.lower()}"
        if dedupe_key in seen_contents:
            continue
        seen_contents.add(dedupe_key)
        title_source = "user" if source == "question" else "assistant"
        items.append(
            {
                "memory_type": memory_type,
                "title": f"{memory_type.title()} from {title_source}"[:80],
                "content": content,
                "keywords": extract_keywords(content),
                "importance": importance,
            }
        )

    trimmed: list[dict] = []
    seen_trimmed: set[str] = set()
    for item in items:
        key = f"{item['memory_type']}::{item['content'].lower()}"
        if key in seen_trimmed:
            continue
        seen_trimmed.add(key)
        trimmed.append(item)
    return trimmed[:6]


def build_workspace_summary(
    workspace_name: str,
    memory_items: list[dict],
    file_items: list[dict],
) -> str:
    lines = [f"Workspace summary for {workspace_name}:"]
    decisions = [row["content"] for row in memory_items if row.get("memory_type") == "decision"][:3]
    tasks = [row["content"] for row in memory_items if row.get("memory_type") == "task"][:3]
    preferences = [row["content"] for row in memory_items if row.get("memory_type") == "preference"][:2]
    facts = [row["content"] for row in memory_items if row.get("memory_type") in {"fact", "summary"}][:2]
    files = [row["title"] for row in file_items[:3]]

    if decisions:
        lines.append("Key decisions: " + " | ".join(decisions))
    if tasks:
        lines.append("Open tasks: " + " | ".join(tasks))
    if preferences:
        lines.append("Preferences: " + " | ".join(preferences))
    if facts:
        lines.append("Recent context: " + " | ".join(facts))
    if files:
        lines.append("Available files: " + " | ".join(files))
    if len(lines) == 1:
        lines.append("No durable workspace memory has been captured yet.")
    return "\n".join(lines)

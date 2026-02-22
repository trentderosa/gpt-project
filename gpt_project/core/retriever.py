import glob
import re
from collections import Counter
from pathlib import Path

from .config import CHUNK_SIZE, KNOWLEDGE_DIR, TOP_K


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    chunks = []
    text = text.strip()
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


def load_knowledge_chunks(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for file_path in glob.glob(str(knowledge_dir / "*.txt")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
        for chunk in chunk_text(text):
            chunks.append((Path(file_path).name, chunk))
    return chunks


def _score_chunk(query_tokens: Counter, chunk: str) -> float:
    chunk_tokens = Counter(_tokenize(chunk))
    if not chunk_tokens:
        return 0.0
    overlap = sum(min(query_tokens[token], chunk_tokens[token]) for token in query_tokens)
    return overlap / (len(query_tokens) + 1)


def retrieve_context(
    query: str, chunks: list[tuple[str, str]], top_k: int = TOP_K
) -> list[tuple[float, str, str]]:
    query_tokens = Counter(_tokenize(query))
    if not query_tokens:
        return []

    scored = []
    for source, chunk in chunks:
        score = _score_chunk(query_tokens, chunk)
        if score > 0:
            scored.append((score, source, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:top_k]


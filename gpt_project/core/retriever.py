import glob
import math
import re
from collections import Counter
from pathlib import Path

from .config import CHUNK_SIZE, KNOWLEDGE_DIR, TOP_K


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9']+", text.lower())


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """Split text at sentence boundaries, keeping chunks under chunk_size chars."""
    text = text.strip()
    if not text:
        return []

    # Split on sentence-ending punctuation followed by whitespace or end-of-string.
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if not sentence:
            continue
        if current and len(current) + 1 + len(sentence) > chunk_size:
            chunks.append(current.strip())
            current = sentence
        else:
            current = (current + " " + sentence).strip() if current else sentence
    if current:
        chunks.append(current.strip())

    # If any sentence alone exceeds chunk_size, hard-split it.
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                final.append(chunk[start:start + chunk_size].strip())
                start += chunk_size
    return [c for c in final if c]


def load_knowledge_chunks(knowledge_dir: Path = KNOWLEDGE_DIR) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    for file_path in glob.glob(str(knowledge_dir / "*.txt")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
        for chunk in chunk_text(text):
            chunks.append((Path(file_path).name, chunk))
    return chunks


class _BM25Index:
    """Lightweight BM25 index built from a list of (source, text) pairs."""

    K1 = 1.5
    B = 0.75

    def __init__(self, chunks: list[tuple[str, str]]) -> None:
        self._chunks = chunks
        self._tokenized: list[list[str]] = [_tokenize(text) for _, text in chunks]
        self._doc_freqs: Counter = Counter()
        for tokens in self._tokenized:
            for token in set(tokens):
                self._doc_freqs[token] += 1
        self._n = len(chunks)
        total_len = sum(len(t) for t in self._tokenized)
        self._avgdl = total_len / max(self._n, 1)

    def _idf(self, token: str) -> float:
        df = self._doc_freqs.get(token, 0)
        return math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query: str, doc_index: int) -> float:
        q_tokens = set(_tokenize(query))
        doc_tokens = Counter(self._tokenized[doc_index])
        dl = len(self._tokenized[doc_index])
        total = 0.0
        for token in q_tokens:
            if token not in doc_tokens:
                continue
            freq = doc_tokens[token]
            idf = self._idf(token)
            numerator = freq * (self.K1 + 1)
            denominator = freq + self.K1 * (1 - self.B + self.B * dl / max(self._avgdl, 1))
            total += idf * numerator / denominator
        return total

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[tuple[float, str, str]]:
        if not query.strip() or not self._chunks:
            return []
        scored = [
            (self.score(query, i), self._chunks[i][0], self._chunks[i][1])
            for i in range(self._n)
        ]
        scored = [(s, src, txt) for s, src, txt in scored if s > 0]
        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[:top_k]


# Module-level index cache keyed by chunk identity.
_index_cache: dict[int, _BM25Index] = {}


def retrieve_context(
    query: str, chunks: list[tuple[str, str]], top_k: int = TOP_K
) -> list[tuple[float, str, str]]:
    cache_key = id(chunks)
    if cache_key not in _index_cache:
        _index_cache[cache_key] = _BM25Index(chunks)
    return _index_cache[cache_key].retrieve(query, top_k=top_k)

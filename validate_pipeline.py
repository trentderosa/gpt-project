"""
Validation script for the Cortex Engine live-data pipeline.
Runs the requested queries through the real ChatService and reports
provider routing, fallback behavior, and answer quality signals.
"""
import logging
import os
import re
import sys
from pathlib import Path


BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from dotenv import load_dotenv

load_dotenv(override=True)


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))

    def flush_lines(self):
        lines = list(self.records)
        self.records.clear()
        return lines


capture = LogCapture()
capture.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(capture)
logging.getLogger().setLevel(logging.DEBUG)


from gpt_project.core.chat_service import ChatService
from gpt_project.core.config import (
    BRAVE_SEARCH_API_KEY,
    DEFAULT_MODEL,
    KNOWLEDGE_DIR,
    TAVILY_API_KEY,
    WEB_SEARCH_PROVIDER,
    WEB_SEARCH_PROVIDER_PRIORITY,
)
from gpt_project.core.llm_wrapper import LLMWrapper
from gpt_project.core.retriever import load_knowledge_chunks
from gpt_project.core.search_tool import build_search_tool


llm = LLMWrapper(model=DEFAULT_MODEL)
chunks = load_knowledge_chunks(KNOWLEDGE_DIR)
search_tool = build_search_tool(
    web_search_provider=WEB_SEARCH_PROVIDER,
    provider_priority=[part.strip() for part in WEB_SEARCH_PROVIDER_PRIORITY.split(",") if part.strip()],
    brave_api_key=BRAVE_SEARCH_API_KEY,
    tavily_api_key=TAVILY_API_KEY,
)

TESTS = [
    ("sports", "what is the best team in the NBA right now"),
    ("sports_event", "who won the super bowl"),
    ("weather", "weather in harrisonburg today"),
    ("finance", "latest nvidia stock price"),
    ("leadership", "who is the CEO of OpenAI"),
]

SEP = "=" * 90


def parse_logs(lines: list[str]) -> dict:
    signals = {
        "category": None,
        "freshness_sensitive": None,
        "source_count": None,
    }
    for line in lines:
        if "ask_start" in line or "ask_stream_start" in line:
            category_match = re.search(r"category=(\S+)", line)
            fresh_match = re.search(r"freshness_sensitive=(\S+)", line)
            if category_match:
                signals["category"] = category_match.group(1)
            if fresh_match:
                signals["freshness_sensitive"] = fresh_match.group(1)
        if "web_search_complete" in line:
            count_match = re.search(r"source_count=(\d+)", line)
            if count_match:
                signals["source_count"] = int(count_match.group(1))
    return signals


def score_answer(query: str, answer: str) -> tuple[str, str]:
    text = answer.lower()
    if any(
        phrase in text
        for phrase in [
            "as of my last update",
            "my knowledge cutoff",
            "i cannot access real-time",
            "unable to retrieve",
            "cannot retrieve",
        ]
    ):
        return "FAIL", "stale/refusal language present"
    if "best team in the nba" in query.lower():
        if re.search(r"\b(celtics?|thunder|cavaliers?|cavs?|nuggets?|knicks?|bucks?|lakers?)\b", text):
            return "PASS", "recognized NBA team named"
        return "WARN", "no clear NBA team named"
    if "super bowl" in query.lower():
        if re.search(r"\b(eagles?|chiefs?|philadelphia|kansas city)\b", text):
            return "PASS", "recognized Super Bowl winner named"
        return "WARN", "no clear winner named"
    if "weather in harrisonburg" in query.lower():
        if re.search(r"\b\d+\b", text) and any(word in text for word in ["forecast", "temperature", "rain", "sun", "cloud", "wind"]):
            return "PASS", "weather details present"
        return "WARN", "weather details look thin"
    if "nvidia" in query.lower():
        if re.search(r"\$[\d,]+\.?\d*|\bnvda\b.*\b\d+\b|\bprice\b.*\b\d+\b", text):
            return "PASS", "price signal present"
        return "WARN", "no clear price found"
    if "ceo of openai" in query.lower():
        if "sam altman" in text:
            return "PASS", "Sam Altman named"
        return "WARN", "Sam Altman not named"
    return "PASS", "no issues detected"


def main() -> None:
    results = []
    print(f"Configured provider mode: {WEB_SEARCH_PROVIDER}")
    print(f"Configured provider priority: {WEB_SEARCH_PROVIDER_PRIORITY}")
    print(SEP)

    for expected_category, question in TESTS:
        svc = ChatService(llm=llm, chunks=chunks, web_search_tool=search_tool)
        capture.records.clear()

        print(f"\n{SEP}")
        print(f"QUERY: {question}")
        print(f"EXPECTED CATEGORY: {expected_category}")
        print(SEP)

        try:
            answer, hits, web_results, _ = svc.ask(
                question=question,
                use_web_search=True,
                user_location_label=None,
                user_timezone=None,
            )
        except Exception as exc:
            print(f"[ERROR] ask() failed: {exc}")
            results.append({"query": question, "status": "FAIL", "reason": str(exc)})
            continue

        log_lines = capture.flush_lines()
        signals = parse_logs(log_lines)
        meta = dict(svc.last_web_search_meta or {})
        status, reason = score_answer(question, answer)

        top_source = "none"
        if web_results:
            top = web_results[0]
            top_source = f"{top.get('url', '?')} | provider={top.get('provider', '?')} | date={top.get('date', '')}"

        print(f"1. CATEGORY ASSIGNED: {signals.get('category')} (expected: {expected_category})")
        print(f"2. FRESHNESS SENSITIVE: {signals.get('freshness_sensitive')}")
        print(f"3. QUERY VARIANTS: {meta.get('query_variants')}")
        print(f"4. PROVIDER SEQUENCE: {meta.get('provider_sequence')}")
        print(f"5. WINNING PROVIDER: {meta.get('winning_provider')}")
        print(f"6. FINAL WINNING SOURCE: {meta.get('winning_source') or top_source}")
        print(f"7. DDG NEEDED: {meta.get('ddg_used')}")
        print(f"8. WEB RESULT COUNT: {signals.get('source_count')}")
        attempts = meta.get("attempts") or []
        for idx, attempt in enumerate(attempts, 1):
            print(
                f"   ATTEMPT {idx}: provider={attempt.get('provider')} stage={attempt.get('stage')} "
                f"fresh={attempt.get('fresh')} accepted={attempt.get('accepted_results')} "
                f"quality={attempt.get('quality_reason')} domains={attempt.get('preferred_domains')} "
                f"query={attempt.get('query')}"
            )
        print("\nFINAL ANSWER")
        print(answer)
        print(f"\n9. CURRENT/CORRECT CHECK: [{status}] {reason}")

        results.append(
            {
                "query": question,
                "status": status,
                "reason": reason,
                "winning_provider": meta.get("winning_provider"),
                "ddg_used": meta.get("ddg_used"),
            }
        )

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    for item in results:
        print(
            f"[{item['status']}] provider={item.get('winning_provider')} ddg={item.get('ddg_used')} "
            f"query={item['query']} -> {item['reason']}"
        )


if __name__ == "__main__":
    main()

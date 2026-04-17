"""
Validation script for the Cortex Engine live-data pipeline.
Runs the required factual/current prompts through the real ChatService and
reports provider routing, fallback behavior, and answer quality.
"""
import logging
import os
import re
import sys
from datetime import datetime
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
    ("sports_event", "who won the super bowl this year"),
    ("sports_event", "who won the masters this year"),
    ("sports", "washington wizards record"),
    ("weather", "weather in harrisonburg today"),
    ("finance", "latest nvidia stock price"),
    ("leadership", "who is the CEO of OpenAI"),
    ("current_events_news", "what happened in the election today"),
    ("finance", "current interest rate"),
    ("product_release_info", "what is the latest iPhone"),
]

SEP = "=" * 90

STALE_PHRASES = [
    "as of my last update",
    "my knowledge cutoff",
    "i cannot access real-time",
    "unable to retrieve",
    "cannot retrieve",
    "don't have real-time",
    "do not have real-time",
    "i don't have access",
    "training data",
    "my training",
    "i was last updated",
    "there are no web results",
    "no web results provided",
]


def parse_logs(lines: list[str]) -> dict:
    signals = {
        "category": None,
        "freshness_sensitive": None,
        "source_count": None,
        "winning_provider": None,
    }
    for line in lines:
        if "ask_start" in line or "ask_stream_start" in line:
            match = re.search(r"category=(\S+)", line)
            if match:
                signals["category"] = match.group(1)
            match = re.search(r"freshness_sensitive=(\S+)", line)
            if match:
                signals["freshness_sensitive"] = match.group(1)
        if "web_search_complete" in line:
            match = re.search(r"source_count=(\d+)", line)
            if match:
                signals["source_count"] = int(match.group(1))
            match = re.search(r"winning_provider=(\S+)", line)
            if match:
                signals["winning_provider"] = match.group(1)
    return signals


def has_stale(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in STALE_PHRASES)


def score_answer(query: str, answer: str, web_results: list[dict], signals: dict) -> tuple[str, str]:
    text = answer.lower()
    source_count = signals.get("source_count") or 0
    q = query.lower()

    if has_stale(answer):
        return "FAIL", "stale/refusal language in answer"

    if any(phrase in text for phrase in ["could not be verified from live sources", "the current answer could not be verified", "cannot verify"]):
        if len(answer.strip()) < 120:
            return "FAIL", "dead-end response with no useful answer"

    if "best team in the nba" in q:
        nba_teams = r"\b(celtics?|thunder|cavaliers?|cavs?|nuggets?|knicks?|bucks?|lakers?|warriors?|suns?|heat|pacers?|hawks?|timberwolves?|grizzlies?|kings?|clippers?|mavericks?|mavs?)\b"
        if re.search(nba_teams, text):
            return "PASS", "NBA team named with live search"
        if source_count > 0 and re.search(r"\b(?:playoff|standings|seeding|seed|bracket|record|regular season)\b", text):
            return "PASS", "live standings/playoff context present"
        return "WARN", "no recognizable NBA team named"

    if "super bowl" in q:
        nfl_teams = r"\b(eagles?|chiefs?|philadelphia|kansas city|ravens?|bills?|49ers?|cowboys?|packers?|steelers?|lions?|bears?|falcons?|saints?|buccaneers?|rams?|seahawks?|bengals?|browns?|patriots?)\b"
        if re.search(nfl_teams, text):
            return "PASS", "Super Bowl winner named with live search"
        return "WARN", "no clear Super Bowl winner named"

    if "masters" in q:
        known_golfers = r"\b(scottie scheffler|rory mcilroy|jon rahm|tiger woods|dustin johnson|brooks koepka|xander schauffele|collin morikawa|viktor hovland|ludvig aberg|bryson dechambeau|patrick cantlay|max homa|tommy fleetwood|adam scott)\b"
        if re.search(known_golfers, text):
            return "PASS", "Masters winner named with live search"
        if re.search(r"(?:won|champion|winner).*masters|masters.*(?:champion|winner)", text):
            return "PASS", "Masters result context present"
        return "WARN", "no clear Masters winner named"

    if "wizards" in q and ("record" in q or "standing" in q or "win" in q):
        if re.search(r"\b\d{1,2}\s*[-–]\s*\d{1,2}\b|\bwins?\b.*\blosses?\b|\blosses?\b.*\bwins?\b", text):
            return "PASS", "W-L record present"
        if "wizards" in text and re.search(r"\b(?:record|standing|wins?|losses?|season|ranked?)\b", text):
            return "WARN", "record context present but no specific numbers"
        return "WARN", "no W-L record or standings context found"

    if "weather in harrisonburg" in q:
        if re.search(r"\b\d+\b", text) and any(word in text for word in ["forecast", "temperature", "degrees", "rain", "sun", "cloud", "wind", "humid", "high", "low"]):
            return "PASS", "weather details with numbers present"
        return "WARN", "weather details look thin"

    if "nvidia" in q and ("stock" in q or "price" in q):
        if re.search(r"\$[\d,]+\.?\d*|\bnvda\b[\s\S]{0,40}\d+|\bprice\b[\s\S]{0,40}\$?\d+", text):
            return "PASS", "price/dollar signal present"
        return "WARN", "no clear Nvidia price found"

    if "ceo of openai" in q:
        if "sam altman" in text:
            return "PASS", "Sam Altman named"
        return "WARN", "Sam Altman not named"

    if "election" in q:
        if source_count > 0 and len(answer) > 80:
            return "PASS", "election query answered with live sources"
        return "WARN", "election answer is thin or vague"

    if "interest rate" in q:
        has_pct = bool(re.search(r"\b\d+\.?\d*\s*%|\b\d+\.?\d*\s*percent", text))
        has_policy_context = bool(
            re.search(r"\bfed(?:eral reserve)?\b|\bfederal funds\b|\bfomc\b|\bpolicy rate\b|\btarget range\b", text)
        )
        if has_pct and has_policy_context:
            return "PASS", "policy/Fed rate percentage stated"
        if has_policy_context and re.search(r"\b(?:cut|hike|hold|raise|lower|target|basis points?|bps|fund)\b", text):
            return "PASS", "Fed rate action/context described"
        if has_pct and re.search(r"\bmortgage|refinance|apr|loan\b", text):
            return "WARN", "consumer/mortgage rate given instead of policy/Fed rate"
        return "WARN", "no specific policy/Fed rate or actionable rate info found"

    if "latest iphone" in q or ("iphone" in q and "latest" in q):
        if re.search(r"\biphone\s*\d{1,2}|\biphone\s+(?:pro|air|plus|max|mini|se)\b", text, re.IGNORECASE):
            return "PASS", "specific iPhone model named"
        return "WARN", "no specific iPhone model named"

    if source_count > 0:
        return "PASS", "live search ran"
    return "WARN", "unclear result"


def main() -> None:
    results = []

    print(SEP)
    print("CORTEX ENGINE PIPELINE VALIDATION")
    print(SEP)
    print(f"Date/Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Provider mode: {WEB_SEARCH_PROVIDER}")
    print(f"Provider priority: {WEB_SEARCH_PROVIDER_PRIORITY}")
    available = search_tool.available_providers() if hasattr(search_tool, "available_providers") else []
    primary = next((provider for provider in available if provider not in {"ddg", "wikipedia"}), "NONE")
    has_premium = primary != "NONE"
    print(f"Available providers: {available}")
    print(f"Primary provider: {primary}")
    if not has_premium:
        print("WARNING: No premium provider (Brave/Tavily) configured. System is DDG-only.")
        print("WARNING: Set BRAVE_SEARCH_API_KEY or TAVILY_API_KEY in .env to enable stronger live search.")
    print(SEP)

    for expected_category, question in TESTS:
        svc = ChatService(llm=llm, chunks=chunks, web_search_tool=search_tool)
        capture.records.clear()

        print(f"\n{SEP}")
        print(f"QUERY    : {question}")
        print(f"EXPECTED : {expected_category}")
        print(SEP)

        try:
            answer, hits, web_results, _ = svc.ask(
                question=question,
                use_web_search=True,
                user_location_label=None,
                user_timezone=None,
            )
        except Exception as exc:
            print(f"[ERROR] ask() raised: {exc}")
            results.append(
                {
                    "query": question,
                    "status": "FAIL",
                    "reason": str(exc),
                    "category": "error",
                    "provider": None,
                    "ddg_used": None,
                    "source_count": 0,
                    "category_ok": False,
                    "expected_category": expected_category,
                }
            )
            continue

        log_lines = capture.flush_lines()
        signals = parse_logs(log_lines)
        meta = dict(svc.last_web_search_meta or {})
        status, reason = score_answer(question, answer, web_results, signals)

        assigned_category = signals.get("category") or "?"
        category_ok = assigned_category == expected_category
        category_flag = "[OK]" if category_ok else f"[WRONG: expected {expected_category}]"

        top_source = "none"
        if web_results:
            top = web_results[0]
            top_source = f"{top.get('url', '?')} | provider={top.get('provider', '?')} | date={top.get('date', '')}"

        print(f"1. CATEGORY       : {assigned_category} {category_flag}")
        print(f"2. FRESHNESS      : {signals.get('freshness_sensitive')}")
        print(f"3. QUERY VARIANTS : {meta.get('query_variants')}")
        print(f"4. PROVIDER SEQ   : {meta.get('provider_sequence')}")
        print(f"5. WINNING PROV   : {meta.get('winning_provider')}")
        print(f"6. TOP SOURCE     : {meta.get('winning_source') or top_source}")
        print(f"7. DDG FALLBACK   : {meta.get('ddg_used')}")
        print(f"8. SOURCE COUNT   : {signals.get('source_count')}")
        for idx, attempt in enumerate(meta.get("attempts") or [], 1):
            print(
                f"   ATTEMPT {idx:02d}: provider={attempt.get('provider')} "
                f"stage={attempt.get('stage')} fresh={attempt.get('fresh')} "
                f"accepted={attempt.get('accepted_results')} "
                f"quality={attempt.get('quality_reason')} "
                f"query={str(attempt.get('query', ''))[:80]}"
            )
        print(f"\nANSWER:\n{answer}")
        print(f"\n9. VERDICT: [{status}] {reason}")

        results.append(
            {
                "query": question,
                "status": status,
                "reason": reason,
                "category": assigned_category,
                "expected_category": expected_category,
                "category_ok": category_ok,
                "provider": meta.get("winning_provider"),
                "ddg_used": meta.get("ddg_used"),
                "source_count": signals.get("source_count") or 0,
            }
        )

    print(f"\n{SEP}")
    print("SUMMARY")
    print(SEP)
    passed = sum(1 for item in results if item["status"] == "PASS")
    warned = sum(1 for item in results if item["status"] == "WARN")
    failed = sum(1 for item in results if item["status"] == "FAIL")
    print(f"PASS={passed}  WARN={warned}  FAIL={failed}  TOTAL={len(results)}")
    print()
    for item in results:
        category_flag = "[OK]" if item.get("category_ok") else f"[!]{item.get('expected_category')}"
        provider = item.get("provider") or "none"
        source_count = item.get("source_count", 0)
        ddg = "ddg" if item.get("ddg_used") else "   "
        print(
            f"[{item['status']:4s}] cat={item['category']:28s}{category_flag}  "
            f"prov={provider:8s} {ddg} src={source_count:2d}  {item['query'][:55]}"
            f"  -> {item['reason']}"
        )
    print()
    if not has_premium:
        print("REMINDER: Premium provider (Brave/Tavily) not configured; some quality limits may be provider-side.")


if __name__ == "__main__":
    main()

"""
End-to-end validation script for the Cortex Engine pipeline.
Runs each test query through the real ChatService and reports routing + result metadata.
"""
import logging
import sys
import os
import json
from datetime import datetime
from io import StringIO
from pathlib import Path

# ── env / path setup ──────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

from dotenv import load_dotenv
load_dotenv(override=True)

# ── capture log output per query ──────────────────────────────────────────────
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

# Attach capture handler before imports that configure loggers
capture = LogCapture()
capture.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(capture)
logging.getLogger().setLevel(logging.DEBUG)

# ── service bootstrap ─────────────────────────────────────────────────────────
from gpt_project.core.config import DEFAULT_MODEL, KNOWLEDGE_DIR
from gpt_project.core.llm_wrapper import LLMWrapper
from gpt_project.core.retriever import load_knowledge_chunks
from gpt_project.core.search_tool import DuckDuckGoSearchTool, CompositeWebSearchTool, WikipediaSearchTool
from gpt_project.core.chat_service import ChatService

llm = LLMWrapper(model=DEFAULT_MODEL)
chunks = load_knowledge_chunks(KNOWLEDGE_DIR)
search_tool = CompositeWebSearchTool(providers=[
    DuckDuckGoSearchTool(),
    WikipediaSearchTool(),
])
svc = ChatService(llm=llm, chunks=chunks, web_search_tool=search_tool)

# ── test queries ──────────────────────────────────────────────────────────────
TESTS = [
    ("sports_event", "who won the super bowl"),
    ("sports_event", "who won the super bowl this year"),
    ("sports_event", "who won the masters this year"),
    ("sports",       "washington wizards record"),
    ("weather",      "weather in harrisonburg today"),
    ("finance",      "latest nvidia stock price"),
    ("leadership",   "who is the CEO of OpenAI"),
]

SEP = "=" * 80

def parse_logs(lines):
    """Extract key routing signals from log lines."""
    signals = {
        "category": None,
        "freshness_sensitive": None,
        "fresh_mode": False,
        "fresh_empty_retry": False,
        "market_filter_relaxed": False,
        "fallback_retry": False,
        "refusal_detected": False,
        "stale_detected": False,
        "notes_suppressed": False,
        "web_source_count": None,
        "query_variants": [],
    }
    for line in lines:
        if "ask_start" in line or "ask_stream_start" in line:
            m_cat = __import__("re").search(r"category=(\S+)", line)
            m_fresh = __import__("re").search(r"freshness_sensitive=(\S+)", line)
            if m_cat:
                signals["category"] = m_cat.group(1)
            if m_fresh:
                signals["freshness_sensitive"] = m_fresh.group(1)
        if "web_search_fresh_mode" in line:
            signals["fresh_mode"] = True
        if "web_search_fresh_empty_retrying" in line:
            signals["fresh_empty_retry"] = True
        if "web_search_market_filter_relaxed" in line:
            signals["market_filter_relaxed"] = True
        if "web_search_fallback_retry" in line:
            signals["fallback_retry"] = True
        if "refusal_output_detected" in line:
            signals["refusal_detected"] = True
        if "stale_answer_detected" in line:
            signals["stale_detected"] = True
        if "notes_suppressed" in line:
            signals["notes_suppressed"] = True
        if "web_search_complete" in line:
            m = __import__("re").search(r"source_count=(\d+)", line)
            if m:
                signals["web_source_count"] = int(m.group(1))
        if "query_classified" in line:
            m = __import__("re").search(r'category=(\S+)\s+question=(.+)$', line)
            if m:
                signals["query_variants"].append(m.group(2).strip("'\""))
    return signals

def score_answer(answer: str, query: str) -> tuple[str, str]:
    """
    Heuristic pass/warn/fail for each known query.
    Returns (status, reason).
    """
    a = answer.lower()
    q = query.lower()
    stale_phrases = [
        "as of 2022", "as of 2023", "as of 2024", "as of early 2025", "as of 2025",
        "as of my last update", "my knowledge cutoff", "my training data",
        "i cannot access real-time", "i don't have access to real-time",
        "unable to retrieve", "cannot retrieve",
    ]
    for p in stale_phrases:
        if p in a:
            return "FAIL", f"stale/refusal phrase present: '{p}'"

    if "super bowl" in q:
        # Should mention a year and a team name
        import re
        if re.search(r"\b(eagle|chief|patriot|ram|buck|bear|49er|bronco|giant|cowboy|charger|dolphin|raider|jet|packer|saint|panther|titan|falcon|seahawk|steeler|bengal|brown|lion|cardinal|raven|texan|colt|jaguar|vike|commander|kansas city|philadelphia|new england|seattle|san francisco|los angeles|green bay|dallas|new york)\b", a):
            return "PASS", "team name found"
        return "WARN", "no recognizable team name in answer"

    if "masters" in q:
        import re
        if re.search(r"\b(scheffler|rahm|rory|mcilroy|woods|johnson|thomas|morikawa|cantlay|spieth|matsuyama|fleetwood|hovland|burns|lower|homa|finau|fitzpatrick|aberg|ludvig)\b", a):
            return "PASS", "golfer name found"
        return "WARN", "no recognizable golfer name"

    if "wizards record" in q:
        import re
        if re.search(r"\b\d+\s*[-–]\s*\d+\b", a):
            return "PASS", "win-loss record pattern found"
        return "WARN", "no win-loss record pattern found"

    if "weather" in q and "harrisonburg" in q:
        import re
        if re.search(r"\b\d+\s*°?\s*[fc]\b|\bfahrenheit\b|\bcelsius\b|\bdegree\b|\btemp\b|\brain\b|\bsunny\b|\bcloudy\b|\bpartly\b|\bforecast\b", a):
            return "PASS", "weather data signal found"
        return "WARN", "no weather data found in answer"

    if "nvidia" in q:
        import re
        if re.search(r"\$[\d,]+\.?\d*|\b\d{2,4}\.?\d*\s*(?:per share|usd|dollars?)\b|\bprice\b.*\b\d+\b|\bnvda\b.*\b\d+\b", a):
            return "PASS", "price signal found"
        return "WARN", "no NVIDIA price value in answer"

    if "ceo of openai" in q:
        if "sam altman" in a:
            return "PASS", "Sam Altman named"
        return "WARN", "Sam Altman not mentioned"

    return "PASS", "no specific checks failed"

# ── run tests ─────────────────────────────────────────────────────────────────
results = []
for expected_cat, question in TESTS:
    # Fresh service instance per test to prevent history contamination
    svc = ChatService(llm=llm, chunks=chunks, web_search_tool=search_tool)
    capture.records.clear()
    print(f"\n{SEP}")
    print(f"QUERY: {question}")
    print(f"EXPECTED CATEGORY: {expected_cat}")
    print(SEP)

    try:
        answer, hits, web_results, profile = svc.ask(
            question=question,
            use_web_search=True,
            user_location_label=None,
            user_timezone=None,
        )
    except Exception as exc:
        print(f"[ERROR] Exception during ask(): {exc}")
        continue

    log_lines = capture.flush_lines()
    signals = parse_logs(log_lines)
    status, reason = score_answer(answer, question)

    # Determine strongest provider evidence
    top_source = "none"
    if web_results:
        top = web_results[0]
        top_source = f"{top.get('url','?')[:70]} | {top.get('date','no date')}"

    print(f"1. CATEGORY ASSIGNED: {signals['category']} (expected: {expected_cat})")
    print(f"2. FRESHNESS_SENSITIVE: {signals['freshness_sensitive']}")
    print(f"3. QUERY VARIANTS LOGGED: {signals['query_variants']}")
    print(f"4. LIVE WEB RAN: {'YES' if signals['web_source_count'] is not None else 'UNKNOWN'} — {signals['web_source_count']} results")
    print(f"5. TOP SOURCE: {top_source}")
    print(f"   FRESH MODE: {signals['fresh_mode']} | FRESH RETRY: {signals['fresh_empty_retry']} | FALLBACK: {signals['fallback_retry']}")
    print(f"6. RETRY/FALLBACK: fresh_empty_retry={signals['fresh_empty_retry']}, market_relaxed={signals['market_filter_relaxed']}, fallback={signals['fallback_retry']}")
    print(f"7. NOTES SUPPRESSED: {signals['notes_suppressed']}")
    print(f"8. REFUSAL GUARD FIRED: {signals['refusal_detected']} | STALE GUARD FIRED: {signals['stale_detected']}")
    print(f"\n--- FINAL ANSWER ---")
    print(answer)
    print(f"\n9. STATUS: [{status}] — {reason}")

    if status != "PASS":
        print("\n*** FAILURE DETAILS ***")
        print("Relevant log lines:")
        for line in log_lines:
            if any(k in line for k in ["ask_start", "web_search", "query_class", "stale", "refusal", "note"]):
                print("  ", line)

    results.append({
        "query": question,
        "expected_cat": expected_cat,
        "assigned_cat": signals["category"],
        "fresh": signals["freshness_sensitive"],
        "web_count": signals["web_source_count"],
        "status": status,
        "reason": reason,
    })

# ── summary table ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"{'Query':<45} {'Cat':>14} {'Fresh':>6} {'Web':>4} {'Result'}")
print("-" * 80)
for r in results:
    cat_match = "OK" if r["assigned_cat"] == r["expected_cat"] else f"X({r['assigned_cat']})"
    print(f"{r['query']:<45} {cat_match:>14} {str(r['fresh']):>6} {str(r['web_count'] or '?'):>4}  [{r['status']}] {r['reason']}")

pass_count = sum(1 for r in results if r["status"] == "PASS")
warn_count = sum(1 for r in results if r["status"] == "WARN")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
print(f"\nTotal: {len(results)} | PASS: {pass_count} | WARN: {warn_count} | FAIL: {fail_count}")

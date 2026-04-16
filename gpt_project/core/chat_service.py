from datetime import datetime, timedelta, timezone
import logging
import re
import xml.etree.ElementTree as ET
import time
import html
from zoneinfo import ZoneInfo
import requests
import csv
from io import StringIO

logger = logging.getLogger("cortex.chat")

from .config import ALWAYS_WEB_SEARCH, MIN_SCORE, TOP_K
from .live_data_store import LiveDataStore
from .llm_wrapper import LLMWrapper
from .retriever import retrieve_context
from .search_tool import DisabledWebSearchTool, WebSearchTool


SYSTEM_PROMPT = """You are Cortex Engine, created by Trent DeRosa.
When asked who created you, say Trent DeRosa.
Answer any user question as helpfully as possible.
Use a direct, candid tone. Do not force uplifting language.
Use correct punctuation, spelling, and capitalization in your responses.
You can mirror the user's tone, but do not mirror punctuation or grammar mistakes.
Use local note context first when relevant.

LIVE WEB RESULTS — PRIMARY SOURCE OF TRUTH:
When web results are provided, they are the authoritative source for factual claims.
Base your answer on those results. Do not contradict or silently ignore web results in favor of training memory.
Cite the most relevant source URLs inline when they directly support a claim.
For any time-sensitive topic — current events, news, prices, sports scores, politics, company details,
software versions, laws, regulations, product info, rankings, schedules, or anything about living people —
rely exclusively on the provided web results. Do not fill gaps from training memory.
If web results are provided but do not clearly answer the question, say so and give a best-effort answer.
If no web results are available and the question is time-sensitive, give your best-effort answer
based on general knowledge and briefly note it may not reflect the latest information.
Never present stale training data as current fact. Never say 'as of my last update' or reference
a past cutoff year as if it is reliable current information.
Never say "unable to retrieve live data", "I don't have real-time access", "I cannot retrieve",
"I cannot access real-time", "I don't have access to live data", or any similar refusal phrase.
Always provide a direct, concrete, useful answer — never a dead end.

If runtime date/time context is provided, use it for questions about today's date, day, or time.
If local notes and web results do not cover a question, use general knowledge and be clear when uncertain.
Never answer with only 'I don't know based on my notes'.
When the user asks for code, respond with a clean markdown code block using the correct language tag and proper indentation.
For simple code requests, keep explanation brief and prioritize readable code/output formatting.
Prefer clear markdown structure when useful: short headings, bullet points, numbered steps, and code fences.
Do not wrap every answer in markdown; choose plain text for short/simple replies.
Use the user profile context to remember the user's name, facts they shared, and mirror their writing style."""

# Regex that identifies queries where live web search adds no value: creative writing,
# translation, rewriting user-supplied text, pure math, and basic small talk.
_WEB_EXEMPT_RE = re.compile(
    r"^(hi|hello|hey|howdy|sup|yo|good morning|good afternoon|good evening|good night|what's up|whats up)\b"
    r"|\bhow are you\b"
    r"|\bwho (are|r) you\b"
    r"|\bwhat is your name\b"
    r"|\b(write|compose|create|generate)\s+(me\s+)?(a\s+|an\s+)?(poem|haiku|sonnet|song|lyrics|story|short story|limerick|fable|fiction|novel|essay|paragraph|narrative|monologue|dialogue|screenplay)\b"
    r"|\b(rewrite|rephrase|improve|fix|edit|proofread|revise|polish|paraphrase|simplify|expand|shorten)\s+(this|the|my|the following|the text|the paragraph|the sentence|below)\b"
    r"|\btranslate\b"
    r"|\bin (french|spanish|german|italian|portuguese|chinese|japanese|korean|arabic|russian|hindi|dutch|swedish|norwegian|danish|turkish|polish|greek)\b",
    re.IGNORECASE,
)

TIME_QUERY_HINTS = {
    "what day",
    "what date",
    "today",
    "time",
    "current time",
    "day is it",
    "date is it",
    "what day is it",
    "what time is it",
    "what year",
    "what season",
    "time of year",
    "current year",
    "current season",
}

NOTES_REFUSAL_HINTS = {
    "i don't know based on my notes",
    "i dont know based on my notes",
    "based on my notes",
    "based on current notes",
}

LIVE_RETRIEVAL_REFUSAL_HINTS = {
    "can't retrieve live",
    "cannot retrieve live",
    "can't pull live",
    "cannot pull live",
    "can't provide live",
    "cannot provide live",
    "unable to retrieve live",
    "unable to retrieve real",
    "i'm unable to retrieve",
    "i am unable to retrieve",
    "don't have real-time",
    "don't have access to real-time",
    "don't have access to live",
    "do not have real-time",
    "do not have access to real-time",
    "can't access real-time",
    "cannot access real-time",
    "can't provide real-time",
    "cannot provide real-time",
    "no access to real-time",
    "no access to live",
    "i'm not able to retrieve",
    "i am not able to retrieve",
    "not able to access live",
    "not able to access real-time",
    "unable to access live",
    "unable to access real-time",
    "can't fetch live",
    "cannot fetch live",
    "unable to fetch live",
    "don't have the ability to retrieve",
}

MARKET_DEFLECTION_HINTS = {
    "live market retrieval is unavailable",
    "market retrieval is unavailable",
    "live retrieval is unavailable",
    "check platforms like",
    "you can check platforms like",
    "you can check yahoo finance",
    "i can't retrieve live stock market data",
    "i cannot retrieve live stock market data",
    "i can't pull live market data",
    "i cannot pull live market data",
    "i can't provide live market data",
    "i cannot provide live market data",
}

WEATHER_QUERY_HINTS = {
    "weather",
    "temperature",
    "forecast",
    "rain",
    "snow",
    "humidity",
    "wind",
    "hot",
    "cold",
}

MARKET_QUERY_HINTS = {
    "stock",
    "stocks",
    "stock market",
    "market",
    "nasdaq",
    "dow",
    "s&p",
    "sp500",
    "spy",
    "qqq",
    "price",
    "share price",
}

NEWS_QUERY_HINTS = {
    "news",
    "headline",
    "headlines",
    "breaking",
    "current events",
    "latest news",
}

SPORTS_QUERY_HINTS = {
    "score",
    "game",
    "match",
    "record",
    "standings",
    "nba",
    "nfl",
    "nhl",
    "mlb",
    "nascar",
    "mls",
    "playoff",
    "playoffs",
    "championship",
    "tournament",
    "league",
    "season",
    "roster",
    "draft",
    "trade",
    "injured",
    "injury",
    "golfer",
    "tennis",
    "masters",
    "super bowl",
    "world series",
    "stanley cup",
    "march madness",
    "world cup",
}

CURRENT_QUERY_HINTS = {
    "today",
    "current",
    "latest",
    "right now",
    "this week",
    "this month",
    "this year",
    "breaking",
    "news",
    "weather",
    "stock",
    "market",
    "price",
    "score",
    "election",
    "bitcoin",
    "crypto",
    "traffic",
    "2025",
    "2026",
    "who won",
    "who is the ceo",
    "who runs",
    "who leads",
}

STALE_ANSWER_HINTS = {
    "as of 2022",
    "as of 2023",
    "as of late 2023",
    "as of early 2024",
    "as of 2024",
    "as of early 2025",
    "as of 2025",
    "as of my last update",
    "up to my last training",
    "based on my training",
    "my knowledge cutoff",
    "my knowledge extends",
    "my training data",
    "i was last updated",
    "as of my training",
    "i don't have the latest specific",
    "i don't have access to real-time",
    "i cannot access real-time",
    # Known stale leadership facts that signal a 2023-era answer
    "emmett shear",         # OpenAI interim CEO for ~4 days in Nov 2023
    "mira murati",          # OpenAI interim CEO briefly in Nov 2023
    "bob chapek",           # Disney CEO fired Nov 2022
    "adam neumann",         # WeWork CEO 2019
    "jack dorsey",          # Twitter CEO, left 2021
}

NAME_PATTERNS = [
    re.compile(r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,30})\b", re.IGNORECASE),
    re.compile(r"\bi am\s+([A-Za-z][A-Za-z\-']{1,30})\b", re.IGNORECASE),
    re.compile(r"\bi'm\s+([A-Za-z][A-Za-z\-']{1,30})\b", re.IGNORECASE),
]

FACT_PATTERNS = [
    re.compile(r"\bi (?:like|love|enjoy)\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi live in\s+(.+)", re.IGNORECASE),
    re.compile(r"\bi go to\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmy major is\s+(.+)", re.IGNORECASE),
    re.compile(r"\bmy favorite\s+(.+)", re.IGNORECASE),
]

ACK_MESSAGES = {
    "ok",
    "okay",
    "k",
    "kk",
    "sounds good",
    "got it",
    "alright",
    "cool",
}

# Authoritative source domains grouped by query category.
_AUTHORITY_DOMAINS: dict[str, set[str]] = {
    "sports": {
        "espn.com", "nfl.com", "nba.com", "mlb.com", "nhl.com", "pga.com",
        "masters.com", "fifa.com", "si.com", "cbssports.com", "bleacherreport.com",
        "olympics.com", "uefa.com", "tennis.com", "atptour.com",
    },
    "finance": {
        "bloomberg.com", "reuters.com", "wsj.com", "marketwatch.com", "cnbc.com",
        "finance.yahoo.com", "sec.gov", "morningstar.com", "ft.com",
    },
    "news": {
        "reuters.com", "apnews.com", "bbc.com", "nytimes.com", "wsj.com",
        "npr.org", "theguardian.com", "politico.com",
    },
    "government": {".gov", ".edu"},
}


class ChatService:
    def __init__(
        self,
        llm: LLMWrapper,
        chunks: list[tuple[str, str]],
        web_search_tool: WebSearchTool | None = None,
    ):
        self.llm = llm
        self.chunks = chunks
        self.history: list[dict] = []
        self.web_search_tool = web_search_tool or DisabledWebSearchTool()
        self.live_store = LiveDataStore()

    def _http_get(self, url: str, params: dict | None = None, timeout: int = 8, attempts: int = 3):
        last_exc = None
        for attempt in range(max(attempts, 1)):
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except Exception as exc:
                last_exc = exc
                if attempt < attempts - 1:
                    time.sleep(0.25 * (attempt + 1))
                    continue
                raise last_exc

    def _needs_runtime_time_context(self, question: str) -> bool:
        q = question.lower()
        return any(hint in q for hint in TIME_QUERY_HINTS)

    def _needs_weather_context(self, question: str) -> bool:
        q = question.lower()
        return any(hint in q for hint in WEATHER_QUERY_HINTS)

    def _needs_market_context(self, question: str) -> bool:
        q = question.lower()
        return any(hint in q for hint in MARKET_QUERY_HINTS)

    def _needs_news_context(self, question: str) -> bool:
        q = question.lower()
        return any(hint in q for hint in NEWS_QUERY_HINTS)

    def _is_current_events_query(self, question: str) -> bool:
        q = question.lower()
        return any(hint in q for hint in CURRENT_QUERY_HINTS)

    def _needs_sports_context(self, question: str) -> bool:
        q = question.lower()
        return any(hint in q for hint in SPORTS_QUERY_HINTS)

    def _is_freshness_sensitive(self, question: str) -> bool:
        """True for queries where only recent search results are trustworthy.
        These queries bypass stale search content by requesting time-limited results.
        """
        return (
            self._is_current_events_query(question)
            or self._needs_sports_context(question)
            or bool(re.search(r'\b(202[5-9]|this year|right now|today|tonight|this week|this month)\b', question, re.IGNORECASE))
        )

    def _is_web_search_exempt(self, question: str) -> bool:
        """Return True for queries where live web search adds no value:
        creative writing, translation, rewriting user text, pure math, small talk."""
        q = question.strip()
        if not q:
            return True
        if q.lower() in ACK_MESSAGES:
            return True
        # Pure arithmetic / math expression
        if re.match(r"^[\d\s\+\-\*\/\^\(\)\.\%=,]+$", q):
            return True
        return bool(_WEB_EXEMPT_RE.search(q))

    # Regex matching major recurring sports championship events.
    _SPORTS_EVENT_RE = re.compile(
        r'\b(?:super bowl|world series|stanley cup|march madness|nba finals?|world cup(?! of)|'
        r'masters(?:\s+(?:tournament|golf))?|pga championship|us open|british open|the open|'
        r'wimbledon|us open tennis|french open|roland garros|australian open|'
        r'olympic games?|olympics?|nfl draft|nba draft|'
        r'ncaa championship|college football playoff|rose bowl|orange bowl|sugar bowl|fiesta bowl)\b',
        re.IGNORECASE,
    )

    def _classify_query(self, question: str) -> str:
        """Return a category label for logging and routing. First match wins."""
        q = question.lower()
        if self._needs_weather_context(question):
            return "weather"
        if self._needs_market_context(question):
            return "finance"
        if self._SPORTS_EVENT_RE.search(q):
            return "sports_event"
        if self._needs_sports_context(question):
            return "sports"
        if re.search(r'\b(?:ceo|president|cfo|cto|head of|founder of|who (?:runs|leads|is the (?:ceo|president|head|chief)))\b', q):
            return "leadership"
        if self._needs_news_context(question):
            return "news"
        if self._is_current_events_query(question):
            return "current_events"
        return "general"

    def _web_queries(self, question: str) -> list[str]:
        query = question.strip()
        if not query:
            return []
        q_lower = query.lower()
        year = datetime.now().year
        clean = re.sub(r'[?!]+$', '', query).strip()
        category = self._classify_query(query)

        logger.info("query_classified category=%s question=%r", category, query[:100])

        # Build a stronger primary query for known factual patterns.
        primary = query

        # Major sports event: super bowl, world series, stanley cup, masters, etc.
        # IMPORTANT: Never use "who won X" as-is — "won" is also the Korean currency (KRW/Won),
        # which causes search engines to return currency exchange results.
        # Always rewrite to "{EventName} {year} winner champion" form.
        if category == "sports_event":
            # Extract the canonical event name from the query.
            event_m = self._SPORTS_EVENT_RE.search(q_lower)
            if event_m:
                event_name = event_m.group(0).title()
                # Use "recap" and "score" to get game result articles, not landing/schedule pages.
                primary = f"{event_name} {year} winner champion score recap"
            else:
                stripped = re.sub(r'^(?:who won(?: the)?|who is winning|winner of|champions? of)\s+', '', clean, flags=re.IGNORECASE).strip()
                primary = f"{stripped} {year} winner champion score recap"

        # Sports: scores, records, standings, game results.
        # Add sport/league context to disambiguate team names ("Washington" → could be many things)
        elif category == "sports":
            # Detect and name the sport to avoid ambiguous team-name matches
            sport_hint = ""
            if re.search(r'\b(?:nba|basketball|lakers?|celtics?|nets?|knicks?|bulls?|heat|bucks?|nuggets?|suns?|warriors?|spurs?|raptors?|wizards?|magic|hawks?|hornets?|pistons?|pacers?|cavs?|cavaliers?|thunder|blazers?|timberwolves?|pelicans?|grizzlies?|jazz|clippers?|mavs?|mavericks?)\b', q_lower):
                sport_hint = "NBA"
            elif re.search(r'\b(?:nfl|football|chiefs?|eagles?|cowboys?|patriots?|ravens?|steelers?|packers?|bears?|lions?|49ers?|rams?|seahawks?|broncos?|chargers?|raiders?|dolphins?|bills?|jets?|texans?|colts?|jaguars?|titans?|bengals?|browns?|giants?|commanders?|cardinals?|saints?|panthers?|falcons?|buccaneers?|vikings?|bears?)\b', q_lower):
                sport_hint = "NFL"
            elif re.search(r'\b(?:mlb|baseball|yankees?|red sox|dodgers?|giants?|mets?|cubs?|astros?|braves?|padres?|phillies?|cardinals?|pirates?|reds?|royals?|brewers?|twins?|tigers?|white sox|guardians?|rockies?|mariners?|athletics?|rangers?|orioles?|blue jays?|rays?|angels?|nationals?)\b', q_lower):
                sport_hint = "MLB"
            elif re.search(r'\b(?:nhl|hockey|rangers?|bruins?|penguins?|capitals?|lightning|blackhawks?|maple leafs?|canadiens?|avalanche|knights?|oilers?|canucks?|flames?|jets?|coyotes?|predators?|blues?|red wings?|sabres?|hurricanes?|blue jackets?|wild|stars?|ducks?|kings?|sharks?|devils?|islanders?|senators?)\b', q_lower):
                sport_hint = "NHL"
            if sport_hint:
                primary = f"{clean} {sport_hint} {year} record standings"
            else:
                primary = f"{clean} {year} latest score result standings"

        # Sports winner/result for non-major-events: strip "who won" to avoid KRW issue.
        elif re.search(r'\b(?:who won|who is winning|who will win|winner of|champions? of|results? of)\b', q_lower):
            stripped = re.sub(r'^(?:who won(?: the)?|who is winning|winner of|champions? of)\s+', '', clean, flags=re.IGNORECASE).strip()
            primary = f"{stripped} {year} winner result"

        # Leadership queries: "who is the CEO of X", "CEO of OpenAI", etc.
        # Use "CEO X" direct form instead of "who is the CEO of X" — cleaner signal for search.
        elif category == "leadership":
            # Extract the org/person from "who is the CEO of X" patterns
            ceo_match = re.search(r'\b(?:ceo|president|cfo|cto|head|founder)\s+(?:of\s+)?([A-Za-z0-9 &.,\-]{2,50})', clean, re.IGNORECASE)
            org = ceo_match.group(1).strip() if ceo_match else clean
            primary = f"{org} CEO current {year}"

        # Weather: let web search results carry the answer when no location coords are available.
        elif category == "weather":
            # Extract location cleanly — strip trailing time words so geocoding gets a clean city name.
            loc_match = re.search(r'\b(?:weather|forecast|temperature)\s+(?:in|for|at)\s+([A-Za-z ,.\-]+?)(?:\s+(?:today|tonight|now|this week|tomorrow|right now))?\s*$', clean, re.IGNORECASE)
            if loc_match:
                loc = loc_match.group(1).strip()
                primary = f"weather {loc} today forecast temperature"
            else:
                primary = f"{clean} weather forecast today"

        # "Latest news on X" or "what happened with X"
        elif re.search(r'\b(?:latest news|what happened|news about|update on|updates on)\b', q_lower):
            primary = f"{clean} {year}"

        # Finance: stock/price queries — keep the original; market context handles structured data.
        # (no rewrite needed; market context fetches direct quotes)

        queries = [primary]

        # For current-event or freshness-sensitive queries, always add year-anchored + "latest" variants.
        if self._is_current_events_query(query) or self._needs_sports_context(query) or primary != query:
            year_query = f"{query} {year}"
            if year_query.lower().strip() != primary.lower().strip():
                queries.append(year_query)
            latest_query = f"latest {query}"
            normed = {q.strip().lower() for q in queries}
            if latest_query.lower().strip() not in normed:
                queries.append(latest_query)

        if self._needs_market_context(query):
            queries.extend(
                [
                    "stock market today S&P 500 Dow Nasdaq",
                    "SPY QQQ DIA market update today",
                ]
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for q in queries:
            norm = q.strip().lower()
            if not norm or norm in seen:
                continue
            seen.add(norm)
            deduped.append(q.strip())
        return deduped[:5]

    def _is_market_result(self, item: dict) -> bool:
        text = " ".join(
            [
                str(item.get("title", "")),
                str(item.get("snippet", "")),
                str(item.get("url", "")),
            ]
        ).lower()
        finance_tokens = {
            "stock",
            "market",
            "s&p",
            "nasdaq",
            "dow",
            "index",
            "shares",
            "equity",
            "finance",
            "invest",
            "spy",
            "qqq",
            "dia",
        }
        return any(token in text for token in finance_tokens)

    def _rank_web_results(self, results: list[dict], question: str) -> list[dict]:
        """Re-rank results to prefer authoritative sources for relevant query types."""
        if len(results) <= 1:
            return results
        q_lower = question.lower()
        authority_domains: set[str] = set()
        if re.search(r'\b(?:won|winner|score|champion|tournament|game|match|season|cup|title)\b', q_lower):
            authority_domains.update(_AUTHORITY_DOMAINS["sports"])
        if re.search(r'\b(?:stock|price|market|nasdaq|dow|share|etf|fund)\b', q_lower):
            authority_domains.update(_AUTHORITY_DOMAINS["finance"])
        if re.search(r'\b(?:ceo|president|founder|leader|executive|who (?:runs|leads))\b', q_lower):
            authority_domains.update(_AUTHORITY_DOMAINS["news"] | _AUTHORITY_DOMAINS["finance"])
        if re.search(r'\b(?:law|regulation|policy|government|election|congress|senate)\b', q_lower):
            authority_domains.update(_AUTHORITY_DOMAINS["government"] | _AUTHORITY_DOMAINS["news"])
        if not authority_domains:
            return results

        def _priority(item: dict) -> int:
            url = (item.get("url") or "").lower()
            for domain in authority_domains:
                if domain in url:
                    return 0
            return 1

        return sorted(results, key=_priority)

    def _build_web_context_block(self, web_results: list[dict]) -> str:
        """Build a structured, prominently-labeled LIVE WEB RESULTS block."""
        if not web_results:
            return ""
        lines = ["=== LIVE WEB RESULTS (PRIMARY SOURCE OF TRUTH) ==="]
        for i, item in enumerate(web_results, 1):
            title = (item.get("title") or "Untitled").strip()
            url = (item.get("url") or "").strip()
            snippet = (item.get("snippet") or "").strip()
            date = (item.get("date") or item.get("published") or item.get("age") or "").strip()
            lines.append(f"\n[Result {i}]")
            lines.append(f"Title: {title}")
            if url:
                lines.append(f"URL: {url}")
            if date:
                lines.append(f"Date: {date}")
            if snippet:
                lines.append(f"Snippet: {snippet}")
        lines.append("\n=== END LIVE WEB RESULTS ===")
        lines.append(
            "\nInstruction: The LIVE WEB RESULTS above are the authoritative source. "
            "Base factual claims on them. Do not contradict or silently override them with training knowledge. "
            "Extract and state specific values directly from the snippets: exact numbers, names, scores, prices, dates, and records. "
            "Prefer concrete facts over vague summaries. "
            "If results are incomplete or conflicting, say so briefly and still give your best-effort answer. "
            "Never say you cannot retrieve live data.\n"
        )
        return "\n".join(lines) + "\n\n"

    def _search_live_web(self, question: str, max_results: int = 5) -> list[dict]:
        merged: list[dict] = []
        seen_urls: set[str] = set()
        market_query = self._needs_market_context(question)
        fresh = self._is_freshness_sensitive(question)

        if fresh:
            logger.info("web_search_fresh_mode question=%r", question[:100])

        def _collect(query: str, use_fresh: bool) -> None:
            try:
                items = self.web_search_tool.search(query, max_results=max_results, fresh=use_fresh)
            except Exception as exc:
                logger.warning("web_search_failed query=%r fresh=%s error=%s", query[:100], use_fresh, exc)
                return
            for item in items:
                url = (item.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                if market_query and not self._is_market_result(item):
                    continue
                seen_urls.add(url)
                merged.append(item)

        # Primary pass: use time-limited search for freshness-sensitive queries.
        for query in self._web_queries(question):
            _collect(query, use_fresh=fresh)
            if len(merged) >= max_results:
                logger.info("web_search_complete fresh=%s source_count=%d", fresh, len(merged))
                return self._rank_web_results(merged[:max_results], question)

        # If fresh-mode returned nothing, retry the same queries WITHOUT the time limit.
        if not merged and fresh:
            logger.info("web_search_fresh_empty_retrying_no_timelimit question=%r", question[:100])
            for query in self._web_queries(question)[:2]:
                _collect(query, use_fresh=False)
                if len(merged) >= max_results:
                    break

        # If market filter dropped everything, retry top query without the filter.
        if not merged and market_query:
            logger.info("web_search_market_filter_relaxed question=%r", question[:100])
            top_query = self._web_queries(question)[0] if self._web_queries(question) else question
            try:
                items = self.web_search_tool.search(top_query, max_results=max_results, fresh=False)
                for item in items:
                    url = (item.get("url") or "").strip()
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        merged.append(item)
            except Exception as exc:
                logger.warning("web_search_market_relaxed_failed error=%s", exc)

        # Final fallback: simplified "latest {question} {year}" query.
        if not merged:
            year = datetime.now().year
            fallback_q = f"latest {question.strip().rstrip('?!')} {year}"
            logger.info("web_search_fallback_retry fallback=%r", fallback_q[:100])
            try:
                items = self.web_search_tool.search(fallback_q, max_results=max_results, fresh=False)
                for item in items:
                    url = (item.get("url") or "").strip()
                    if url and url not in seen_urls:
                        merged.append(item)
            except Exception as exc:
                logger.warning("web_search_fallback_failed error=%s", exc)

        logger.info("web_search_complete fresh=%s source_count=%d", fresh, len(merged))
        return self._rank_web_results(merged, question)

    def _trim_history(self, history: list[dict], max_messages: int = 16, max_chars: int = 12000) -> list[dict]:
        trimmed = history[-max_messages:]
        total_chars = sum(len(msg.get("content", "")) for msg in trimmed)
        while trimmed and total_chars > max_chars:
            total_chars -= len(trimmed[0].get("content", ""))
            trimmed.pop(0)
        return trimmed

    def _extract_name(self, text: str) -> str | None:
        clean = text.strip()
        for pattern in NAME_PATTERNS:
            match = pattern.search(clean)
            if match:
                name = match.group(1).strip(" .,!?:;")
                if 2 <= len(name) <= 31:
                    return name
        return None

    def _extract_facts(self, text: str) -> list[str]:
        clean = text.strip()
        facts: list[str] = []
        for pattern in FACT_PATTERNS:
            match = pattern.search(clean)
            if match:
                fact = match.group(0).strip(" .,!?:;")
                if len(fact) >= 6:
                    facts.append(fact)
        return facts

    def _infer_style(self, messages: list[dict]) -> dict:
        user_messages = [m.get("content", "") for m in messages if m.get("role") == "user"]
        if not user_messages:
            return {}
        tail = user_messages[-20:]
        joined = " ".join(tail)
        char_count = max(len(joined), 1)
        lower_ratio = sum(1 for c in joined if c.islower()) / char_count
        exclaim_count = joined.count("!")
        question_count = joined.count("?")
        avg_len = sum(len(msg.split()) for msg in tail) / max(len(tail), 1)
        slang_tokens = {"bruh", "bro", "yo", "fr", "nah", "idk", "lmao", "lol"}
        token_count = max(len(joined.split()), 1)
        slang_count = sum(1 for token in joined.lower().split() if token.strip(".,!?") in slang_tokens)

        vibe = "neutral"
        if slang_count / token_count > 0.05:
            vibe = "casual"
        if lower_ratio > 0.90 and avg_len < 10:
            vibe = "brief"
        if avg_len > 25:
            vibe = "detailed"

        return {
            "vibe": vibe,
            "lowercase_ratio": round(lower_ratio, 2),
            "uses_exclamation": exclaim_count > question_count and exclaim_count >= 2,
            "avg_words_per_message": round(avg_len, 1),
        }

    def _merge_user_profile(self, existing: dict, question: str, history: list[dict]) -> dict:
        profile = dict(existing or {})
        name = self._extract_name(question)
        if name:
            profile["name"] = name

        existing_facts = profile.get("facts", [])
        if not isinstance(existing_facts, list):
            existing_facts = []
        facts = self._extract_facts(question)
        seen = {str(item).lower() for item in existing_facts}
        for fact in facts:
            if fact.lower() not in seen:
                existing_facts.append(fact)
                seen.add(fact.lower())
        profile["facts"] = existing_facts[-20:]
        profile["style"] = self._infer_style(history + [{"role": "user", "content": question}])
        return profile

    def _profile_context_block(self, profile: dict) -> str:
        if not profile:
            return ""
        lines = ["User profile context:"]
        name = profile.get("name")
        if name:
            lines.append(f"- Name: {name}")
        facts = profile.get("facts") or []
        if facts:
            lines.append("- Known user facts:")
            for fact in facts[-8:]:
                lines.append(f"  - {fact}")
        style = profile.get("style") or {}
        if style:
            lines.append(
                f"- Style preference hint: vibe={style.get('vibe', 'neutral')}, "
                f"avg_words={style.get('avg_words_per_message', 'n/a')}, "
                f"lowercase_ratio={style.get('lowercase_ratio', 'n/a')}"
            )
        return "\n".join(lines) + "\n\n"

    def _looks_like_notes_refusal(self, answer: str) -> bool:
        text = answer.lower()
        return any(hint in text for hint in NOTES_REFUSAL_HINTS)

    def _looks_like_live_retrieval_refusal(self, answer: str) -> bool:
        text = (answer or "").lower()
        return any(hint in text for hint in LIVE_RETRIEVAL_REFUSAL_HINTS)

    def _looks_like_market_deflection(self, answer: str) -> bool:
        text = (answer or "").lower()
        return any(hint in text for hint in MARKET_DEFLECTION_HINTS) or self._looks_like_live_retrieval_refusal(
            answer
        )

    def _looks_like_refusal_output(self, answer: str) -> bool:
        """True if the answer contains any live-retrieval refusal or stale-data phrase."""
        return self._looks_like_live_retrieval_refusal(answer) or self._looks_stale_current_answer(answer)

    def _looks_stale_current_answer(self, answer: str) -> bool:
        text = (answer or "").lower()
        return any(hint in text for hint in STALE_ANSWER_HINTS)

    def _normalize_response_punctuation(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return "I could not generate a response."

        # Convert HTML-style code blocks to markdown fences.
        known_langs = {
            "python", "py", "javascript", "js", "typescript", "ts", "bash", "sh", "shell",
            "json", "sql", "html", "css", "java", "c", "cpp", "c++", "csharp", "cs", "go", "rust",
        }

        def _html_code_to_markdown(match: re.Match) -> str:
            raw = html.unescape(match.group(1) or "").strip()
            if not raw:
                return "```\n```"
            lines = raw.splitlines()
            first = lines[0].strip().lower() if lines else ""
            if first in known_langs and len(lines) > 1:
                lang = "python" if first == "py" else ("javascript" if first == "js" else first)
                body = "\n".join(lines[1:]).strip("\n")
                return f"```{lang}\n{body}\n```"
            return f"```\n{raw}\n```"

        cleaned = re.sub(
            r"<pre>\s*<code[^>]*>([\s\S]*?)</code>\s*</pre>",
            _html_code_to_markdown,
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = html.unescape(cleaned)

        # Protect fenced code blocks from punctuation/capitalization rewrites.
        code_blocks = re.findall(r"```[\s\S]*?```", cleaned)
        for i, block in enumerate(code_blocks):
            token = f"__CODE_BLOCK_{i}__"
            cleaned = cleaned.replace(block, token, 1)

        # Normalize spacing.
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned).strip()
        cleaned = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\bi\b", "I", cleaned)

        has_markdown_layout = bool(
            re.search(r"(?m)^\s*(?:[-*]\s+|\d+\.\s+|#{1,6}\s+|>\s+)", cleaned)
        )
        if not has_markdown_layout:
            # Capitalize sentence starts for plain text responses.
            parts = re.split(r"([.!?]\s+)", cleaned)
            rebuilt = []
            for part in parts:
                if not part:
                    continue
                if re.match(r"[.!?]\s+", part):
                    rebuilt.append(part)
                    continue
                segment = list(part)
                for i, ch in enumerate(segment):
                    if ch.isalpha():
                        segment[i] = ch.upper()
                        break
                rebuilt.append("".join(segment))
            cleaned = "".join(rebuilt).strip()

            # Ensure terminal punctuation when response ends in alphanumeric text.
            if cleaned[-1].isalnum():
                cleaned += "."

        # Restore code blocks unchanged.
        for i, block in enumerate(code_blocks):
            cleaned = cleaned.replace(f"__CODE_BLOCK_{i}__", block)
        return cleaned

    def _runtime_time_context(
        self,
        user_timezone: str | None = None,
        user_utc_offset_minutes: int | None = None,
        user_location_label: str | None = None,
    ) -> str:
        now: datetime
        zone_label = "server-local"

        if user_timezone:
            try:
                now = datetime.now(ZoneInfo(user_timezone))
                zone_label = user_timezone
            except Exception:
                now = datetime.now()
        elif user_utc_offset_minutes is not None:
            tz = timezone(-timedelta(minutes=user_utc_offset_minutes))
            now = datetime.now(tz)
            zone_label = f"UTC offset {(-user_utc_offset_minutes / 60):+g}h"
        else:
            now = datetime.now()

        location_line = f"- User location hint: {user_location_label}\n" if user_location_label else ""
        season = self._season_for_month(
            month=now.month,
            hemisphere="south" if self._infer_hemisphere(user_location_label) == "south" else "north",
        )
        return (
            f"Runtime context:\n"
            f"- Local datetime: {now.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"- Day of week: {now.strftime('%A')}\n"
            f"- Current year: {now.year}\n"
            f"- Current season: {season}\n"
            f"- Timezone used: {zone_label}\n"
            f"{location_line}\n"
        )

    def _infer_hemisphere(self, user_location_label: str | None) -> str:
        if not user_location_label:
            return "north"
        match = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", user_location_label)
        if not match:
            return "north"
        lat = float(match.group(1))
        return "south" if lat < 0 else "north"

    def _season_for_month(self, month: int, hemisphere: str = "north") -> str:
        if hemisphere == "south":
            # Shift seasons by 6 months
            month = ((month + 5) % 12) + 1
        if month in (12, 1, 2):
            return "winter"
        if month in (3, 4, 5):
            return "spring"
        if month in (6, 7, 8):
            return "summer"
        return "fall"

    def _parse_lat_lon(self, user_location_label: str | None) -> tuple[float, float] | None:
        if not user_location_label:
            return None
        match = re.match(r"\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$", user_location_label)
        if not match:
            return None
        return float(match.group(1)), float(match.group(2))

    def _extract_weather_location_from_question(self, question: str) -> str | None:
        q = (question or "").strip()
        if not q:
            return None
        # Strip trailing time qualifiers before matching location.
        q_clean = re.sub(r'\s+(?:today|tonight|now|right now|this week|this morning|this afternoon|tomorrow|currently)\s*$', '', q, flags=re.IGNORECASE).strip()
        patterns = [
            re.compile(r"\b(?:weather|temp(?:erature)?|forecast)\s+(?:in|for|at)\s+([A-Za-z0-9.,\-\s]{2,80?}?)(?:\s+(?:today|tonight|now|right now|this week|tomorrow))?\s*$", re.IGNORECASE),
            re.compile(r"\b(?:weather|temp(?:erature)?|forecast)\s+(?:in|for|at)\s+([A-Za-z0-9.,\-\s]{2,80})\s*$", re.IGNORECASE),
            re.compile(r"\bin\s+([A-Za-z0-9.,\-\s]{2,80})\s*$", re.IGNORECASE),
        ]
        for pattern in patterns:
            for candidate in (q_clean, q):
                match = pattern.search(candidate)
                if match:
                    location = match.group(1).strip(" .,!?:;")
                    # Strip any trailing time qualifiers that slipped through
                    location = re.sub(r'\s+(?:today|tonight|now|right now|this week|this morning|this afternoon|tomorrow|currently)\s*$', '', location, flags=re.IGNORECASE).strip(" .,!?:;")
                    if len(location) >= 2:
                        return location
        return None

    def _geocode_location(self, location_text: str) -> tuple[float, float] | None:
        if not location_text:
            return None
        try:
            resp = self._http_get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location_text, "count": 1, "language": "en", "format": "json"},
                timeout=8,
                attempts=3,
            )
            payload = resp.json()
            results = payload.get("results") or []
            if not results:
                return None
            top = results[0]
            lat = top.get("latitude")
            lon = top.get("longitude")
            if lat is None or lon is None:
                return None
            return float(lat), float(lon)
        except Exception:
            return None

    def _use_us_units(self, user_location_label: str | None, user_timezone: str | None) -> bool:
        lat_lon = self._parse_lat_lon(user_location_label)
        if lat_lon:
            lat, lon = lat_lon
            in_contiguous_us = 24.0 <= lat <= 49.6 and -125.0 <= lon <= -66.0
            in_alaska = 51.0 <= lat <= 72.0 and -170.0 <= lon <= -129.0
            in_hawaii = 18.5 <= lat <= 22.5 and -161.0 <= lon <= -154.0
            in_puerto_rico = 17.5 <= lat <= 18.7 and -67.5 <= lon <= -65.0
            if in_contiguous_us or in_alaska or in_hawaii or in_puerto_rico:
                return True
        # Fallback when coordinates are unavailable.
        return bool(user_timezone and user_timezone.startswith("America/"))

    def _weather_context(
        self,
        user_location_label: str | None,
        user_timezone: str | None = None,
        question: str | None = None,
    ) -> str:
        lat_lon = self._parse_lat_lon(user_location_label)
        if not lat_lon:
            query_location = self._extract_weather_location_from_question(question or "")
            if query_location:
                lat_lon = self._geocode_location(query_location)
        if not lat_lon:
            # No location available — let web search results cover the weather question.
            return ""
        lat, lon = lat_lon
        use_us_units = self._use_us_units(
            user_location_label=user_location_label,
            user_timezone=user_timezone,
        )
        temperature_unit = "fahrenheit" if use_us_units else "celsius"
        wind_unit = "mph" if use_us_units else "kmh"
        temp_label = "F" if use_us_units else "C"
        wind_label = "mph" if use_us_units else "km/h"
        try:
            resp = self._http_get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": "temperature_2m,apparent_temperature,weather_code,wind_speed_10m",
                    "timezone": "auto",
                    "temperature_unit": temperature_unit,
                    "wind_speed_unit": wind_unit,
                },
                timeout=8,
                attempts=3,
            )
            payload = resp.json()
            cur = payload.get("current", {})
            return (
                "Live weather context:\n"
                f"- Coordinates: {lat:.3f}, {lon:.3f}\n"
                f"- Observation time: {cur.get('time', 'unknown')}\n"
                f"- Temperature {temp_label}: {cur.get('temperature_2m', 'unknown')}\n"
                f"- Feels like {temp_label}: {cur.get('apparent_temperature', 'unknown')}\n"
                f"- Wind {wind_label}: {cur.get('wind_speed_10m', 'unknown')}\n"
                f"- Weather code: {cur.get('weather_code', 'unknown')}\n\n"
            )
        except Exception:
            logger.warning("weather_api_failed lat=%s lon=%s", lat, lon)
            return ""

    # Company name → ticker mapping for when users type full company names instead of tickers.
    _COMPANY_TO_TICKER: dict[str, str] = {
        "nvidia": "NVDA",
        "apple": "AAPL",
        "microsoft": "MSFT",
        "tesla": "TSLA",
        "amazon": "AMZN",
        "google": "GOOGL",
        "alphabet": "GOOGL",
        "meta": "META",
        "netflix": "NFLX",
        "paypal": "PYPL",
        "salesforce": "CRM",
        "amd": "AMD",
        "intel": "INTC",
        "qualcomm": "QCOM",
        "broadcom": "AVGO",
        "arm": "ARM",
        "palantir": "PLTR",
        "coinbase": "COIN",
        "berkshire": "BRK",
        "jpmorgan": "JPM",
        "jp morgan": "JPM",
        "goldman sachs": "GS",
        "bank of america": "BAC",
        "s&p 500": "SPY",
        "s&p500": "SPY",
        "dow jones": "DIA",
        "nasdaq": "QQQ",
    }

    def _extract_symbols_from_question(self, question: str) -> list[str]:
        """Pick uppercase ticker-style tokens, $-prefixed symbols, common tickers, and company names."""
        dollar_tokens = [m.upper() for m in re.findall(r"\$([A-Za-z]{1,5})\b", question)]
        candidates = re.findall(r"\b[A-Z]{1,5}\b", question)
        lowered = question.lower()
        common = [
            "spy", "qqq", "dia", "iwm", "aapl", "msft", "nvda", "tsla",
            "amzn", "meta", "googl", "gspc", "dji", "ixic", "amd", "intc",
            "arm", "pltr", "nflx", "coin",
        ]
        common_hits = [sym.upper() for sym in common if re.search(rf"\b{re.escape(sym)}\b", lowered)]
        # Company name → ticker mappings
        company_hits = []
        for name, ticker in self._COMPANY_TO_TICKER.items():
            if name in lowered:
                company_hits.append(ticker)
        deny = {"I", "A", "AN", "THE", "AND", "OR", "IT", "WE", "YOU"}
        symbols = [c for c in dollar_tokens + candidates + common_hits + company_hits if c not in deny]
        deduped: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            deduped.append(symbol)
        return deduped[:8]

    def _live_market_context(self, question: str) -> str:
        default_symbols = ["SPY", "QQQ", "DIA", "IWM", "AAPL", "MSFT", "NVDA", "TSLA", "GSPC", "DJI", "IXIC"]
        symbols = self._extract_symbols_from_question(question) or default_symbols
        lines: list[str] = []

        # Source 1: Yahoo chart meta per symbol (reliable for ETFs, indices, and US equities).
        index_map = {
            "SPY": "SPY",
            "QQQ": "QQQ",
            "DIA": "DIA",
            "IWM": "IWM",
            "GSPC": "%5EGSPC",
            "DJI": "%5EDJI",
            "IXIC": "%5EIXIC",
        }
        yahoo_lines: list[str] = []
        for symbol in symbols[:10]:
            key = symbol.upper().lstrip("^")
            ticker = index_map.get(key, symbol.upper())
            try:
                resp = self._http_get(
                    f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                    params={"range": "1d", "interval": "5m"},
                    timeout=8,
                    attempts=2,
                )
                payload = resp.json()
                result = ((payload.get("chart") or {}).get("result") or [])
                if not result:
                    continue
                meta = result[0].get("meta") or {}
                price = meta.get("regularMarketPrice")
                prev = meta.get("previousClose")
                shown_symbol = str(meta.get("symbol") or symbol).upper()
                if price is None:
                    continue
                change_txt = ""
                if isinstance(prev, (int, float)) and prev:
                    change_pct = ((float(price) - float(prev)) / float(prev)) * 100.0
                    change_txt = f", change={change_pct:+.2f}%"
                yahoo_lines.append(f"- {shown_symbol}: price={price}{change_txt}")
            except Exception:
                continue
        if yahoo_lines:
            lines = ["Live market context (latest quotes):"] + yahoo_lines

        # Source 2: Stooq per symbol fallback.
        if len(lines) <= 1:
            stooq_lines: list[str] = []
            for symbol in symbols[:10]:
                key = symbol.upper().lstrip("^")
                if key in {"GSPC", "DJI", "IXIC"}:
                    continue
                stooq_ticker = f"{key.lower()}.us"
                try:
                    resp = self._http_get(
                        f"https://stooq.com/q/l/?s={stooq_ticker}&f=sd2t2ohlcv&h&e=csv",
                        timeout=8,
                        attempts=2,
                    )
                    reader = csv.DictReader(StringIO(resp.text))
                    rows = list(reader)
                    if not rows:
                        continue
                    row = rows[0]
                    close = str(row.get("Close") or "").strip()
                    if not re.match(r"^-?\d+(\.\d+)?$", close):
                        continue
                    shown_symbol = str(row.get("Symbol") or key).upper()
                    date = str(row.get("Date") or "N/A").strip()
                    time_val = str(row.get("Time") or "N/A").strip()
                    volume = str(row.get("Volume") or "N/A").strip()
                    stooq_lines.append(
                        f"- {shown_symbol}: close={close}, date={date}, time={time_val}, volume={volume}"
                    )
                except Exception:
                    continue
            if stooq_lines:
                lines = ["Live market context (latest quotes):"] + stooq_lines

        if lines:
            return "\n".join(lines) + "\n\n"

        rows = self.live_store.get_latest("stock", limit=10)
        if not rows:
            logger.info("market_lookup_all_sources_failed question=%r", question[:100])
            return ""
        lines = ["Live market context (cached snapshot):"]
        for item in rows[:10]:
            payload = item.get("payload", {})
            symbol = str(payload.get("symbol", item.get("source_key", "unknown"))).upper()
            close = payload.get("close", "N/A")
            date = payload.get("date", item.get("created_at", "N/A"))
            time_val = payload.get("time", "N/A")
            volume = payload.get("volume", "N/A")
            lines.append(
                f"- {symbol}: close={close}, date={date}, time={time_val}, volume={volume}"
            )
        return "\n".join(lines) + "\n\n"

    def _live_news_context(self, max_age_hours: int = 12) -> str:
        rows = self.live_store.get_latest("news", limit=5)
        if not rows:
            return ""
        now = datetime.now(timezone.utc)
        lines = ["Live news context (latest headlines):"]
        any_fresh = False
        for row in rows[:5]:
            # Skip rows whose created_at is older than max_age_hours.
            created_at_raw = (row.get("created_at") or "").strip()
            if created_at_raw:
                try:
                    created_at = datetime.fromisoformat(created_at_raw)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    age_hours = (now - created_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        logger.info("live_news_stale_skipped age_hours=%.1f", age_hours)
                        continue
                except Exception:
                    pass  # If we can't parse the date, include it anyway
            payload = row.get("payload", {})
            items = payload.get("items", [])
            for item in items[:4]:
                title = (item.get("title") or "").strip() or "Untitled"
                link = (item.get("link") or "").strip()
                pub_date = (item.get("pub_date") or "").strip()
                lines.append(f"- {title} ({pub_date}) {link}")
                any_fresh = True
        if not any_fresh:
            return ""
        return "\n".join(lines) + "\n\n"

    def _extract_market_lines(self, market_context: str, prefer_symbols: list[str] | None = None) -> list[str]:
        """Return market data lines from context, with requested symbols surfaced first."""
        all_lines = []
        for line in (market_context or "").splitlines():
            clean = line.strip()
            if clean.startswith("- ") and ("price=" in clean or "close=" in clean):
                all_lines.append(clean)
        if not prefer_symbols or not all_lines:
            return all_lines[:5]
        # Partition: requested symbols first, then the rest.
        wanted = [ln for ln in all_lines if any(sym.upper() in ln.upper() for sym in prefer_symbols)]
        others = [ln for ln in all_lines if ln not in wanted]
        return (wanted + others)[:5]

    def _answer_contains_market_snapshot(self, answer: str) -> bool:
        text = (answer or "").lower()
        # Match ticker:close= format (from our market context lines)
        if "latest market snapshot" in text:
            return True
        if re.search(r"\b[a-z0-9\^]{1,6}\s*:\s*(price|close)\s*=", text):
            return True
        # Also match prose price mentions: "$197.92", "197.92 per share", "trading at 197", etc.
        if re.search(r"\$[\d,]+\.?\d{0,2}\b", text):
            return True
        if re.search(r"\btrading\s+at\s+[\d,]+\.?\d{0,2}\b", text):
            return True
        if re.search(r"\bprice\s+(?:is|of|at|:)\s+\$?[\d,]+\.?\d{0,2}\b", text):
            return True
        return False

    def _market_answer_fallback(self, market_context: str, web_results: list[dict], question: str = "") -> str:
        requested = self._extract_symbols_from_question(question) if question else []
        market_lines = self._extract_market_lines(market_context, prefer_symbols=requested)
        if market_lines:
            body = "\n".join(market_lines[:4])
            return f"Here is the latest market snapshot I found:\n{body}\n\nAsk for a specific ticker if you want a deeper breakdown."
        if web_results:
            titles = [item.get("title", "").strip() for item in web_results[:3] if item.get("title")]
            if titles:
                joined = "; ".join(titles)
                return (
                    "I could not pull structured quotes this second, but current market sources are reporting updates now: "
                    f"{joined}. Ask for SPY, QQQ, or a specific ticker for a focused update."
                )
        return (
            "I cannot confirm exact live quotes at this second, but I can still help with a best-effort market read. "
            "If you share a ticker (for example SPY, QQQ, AAPL, or TSLA), I will give a focused breakdown and what to watch next."
        )

    def _topic_news_context(self, question: str) -> str:
        # Direct topic RSS fallback for "latest news on X" style questions.
        q = question.strip()
        if not q:
            return ""
        try:
            resp = self._http_get(
                "https://news.google.com/rss/search",
                params={"q": q, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                timeout=8,
                attempts=2,
            )
            root = ET.fromstring(resp.content)
            items = root.findall(".//item")
            if not items:
                return ""
            lines = ["Live topic news context:"]
            for item in items[:8]:
                title = (item.findtext("title") or "").strip() or "Untitled"
                link = (item.findtext("link") or "").strip()
                pub_date = (item.findtext("pubDate") or "").strip()
                lines.append(f"- {title} ({pub_date}) {link}")
            return "\n".join(lines) + "\n\n"
        except Exception:
            return ""

    def ask(
        self,
        question: str,
        history: list[dict] | None = None,
        use_web_search: bool = True,
        user_timezone: str | None = None,
        user_utc_offset_minutes: int | None = None,
        user_location_label: str | None = None,
        user_profile: dict | None = None,
        uploaded_file_context: str | None = None,
        model: str | None = None,
    ) -> tuple[str, list[tuple[float, str, str]], list[dict], dict]:
        normalized_question = question.strip().lower()
        if normalized_question in ACK_MESSAGES:
            short_reply = "Okay."
            effective_history = history if history is not None else self.history
            updated_profile = self._merge_user_profile(user_profile or {}, question, effective_history)
            if history is None:
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": short_reply})
            return short_reply, [], [], updated_profile

        category = self._classify_query(question)
        is_fresh = self._is_freshness_sensitive(question)
        logger.info("ask_start category=%s freshness_sensitive=%s question=%r", category, is_fresh, question[:100])

        hits = retrieve_context(question, self.chunks, top_k=TOP_K)
        has_strong_note_context = bool(hits and hits[0][0] >= MIN_SCORE)
        if has_strong_note_context:
            context = "\n\n".join(
                f"[source={source} score={score:.3f}]\n{text}" for score, source, text in hits
            )
        else:
            context = ""

        web_results: list[dict] = []
        _effective_web = use_web_search or (ALWAYS_WEB_SEARCH and not self._is_web_search_exempt(question))
        if _effective_web:
            web_results = self._search_live_web(question, max_results=5)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        effective_history = history if history is not None else self.history
        updated_profile = self._merge_user_profile(user_profile or {}, question, effective_history)
        profile_context = self._profile_context_block(updated_profile)
        messages.extend(self._trim_history(effective_history))
        runtime_context = ""
        if self._needs_runtime_time_context(question) or self._is_current_events_query(question) or self._is_freshness_sensitive(question):
            runtime_context = (
                self._runtime_time_context(
                    user_timezone=user_timezone,
                    user_utc_offset_minutes=user_utc_offset_minutes,
                    user_location_label=user_location_label,
                )
            )
        weather_context = ""
        if self._needs_weather_context(question):
            weather_context = self._weather_context(
                user_location_label=user_location_label,
                user_timezone=user_timezone,
                question=question,
            )
        market_context = ""
        if self._needs_market_context(question):
            market_context = self._live_market_context(question=question)
        news_context = ""
        if self._needs_news_context(question):
            news_context = self._live_news_context()
            topic_news = self._topic_news_context(question)
            if topic_news:
                news_context += topic_news
        web_context = self._build_web_context_block(web_results)
        note_context_block = ""
        # Suppress BM25 note context for freshness-sensitive queries when live web results exist.
        # Local notes are static and can inject outdated facts that compete with live search results.
        if context and not (self._is_freshness_sensitive(question) and web_results):
            note_context_block = f"Context from local notes:\n{context}\n\n"
        elif context and self._is_freshness_sensitive(question) and web_results:
            logger.info("notes_suppressed_freshness_sensitive category=%s web_results=%d", category, len(web_results))

        has_live_signal = bool(web_results) or bool(market_context.strip()) or bool(news_context.strip()) or bool(weather_context.strip())
        anti_refusal = "" if has_strong_note_context else "Instruction: If local notes do not cover this, answer from general knowledge.\n\n"
        anti_stale = "Instruction: Use the live context provided above. Do not reference training cutoff dates.\n\n" if self._is_freshness_sensitive(question) and has_live_signal else ""

        messages.append(
            {
                "role": "user",
                "content": (
                    f"{runtime_context}"
                    f"{weather_context}"
                    f"{market_context}"
                    f"{news_context}"
                    f"{web_context}"
                    f"{profile_context}"
                    f"{uploaded_file_context or ''}"
                    f"{note_context_block}"
                    f"{anti_refusal}"
                    f"{anti_stale}"
                    f"User question:\n{question}"
                ),
            }
        )

        answer = self.llm.chat(messages=messages, model=model)
        if self._looks_like_notes_refusal(answer):
            fallback_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            fallback_messages.extend(self._trim_history(effective_history))
            fallback_messages.append(
                {
                    "role": "user",
                    "content": (
                        f"{runtime_context}"
                        f"{weather_context}"
                        f"{market_context}"
                        f"{news_context}"
                        f"{web_context}"
                        f"{profile_context}"
                        f"{uploaded_file_context or ''}"
                        f"Question: {question}\n\n"
                        "Give a direct answer using available live context or general knowledge. "
                        "If uncertain, state uncertainty briefly, then still provide your best useful answer."
                    ),
                }
            )
            answer = self.llm.chat(messages=fallback_messages, model=model)

        # Universal refusal guard: if the model still produced a live-retrieval refusal, regenerate once.
        if self._looks_like_refusal_output(answer):
            logger.info("refusal_output_detected question=%r — regenerating", question[:100])
            regen_msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
            regen_msgs.extend(self._trim_history(effective_history))
            regen_msgs.append(
                {
                    "role": "user",
                    "content": (
                        f"{runtime_context}"
                        f"{weather_context}"
                        f"{market_context}"
                        f"{news_context}"
                        f"{web_context}"
                        f"{profile_context}"
                        f"{uploaded_file_context or ''}"
                        f"Question: {question}\n\n"
                        "Give a direct, concrete, useful answer. "
                        "Do not say you cannot access real-time or live data. "
                        "Use the live context above if available. Otherwise use general knowledge and note uncertainty briefly."
                    ),
                }
            )
            answer = self.llm.chat(messages=regen_msgs, model=model)

        if (self._is_current_events_query(question) or self._needs_sports_context(question)) and self._looks_stale_current_answer(answer):
            logger.info("stale_answer_detected question=%r — rerunning with web context", question[:100])
            has_live_signal = bool(web_results) or bool(market_context.strip()) or bool(news_context.strip()) or bool(
                weather_context.strip()
            )
            if not has_live_signal:
                if self._needs_market_context(question):
                    answer = self._market_answer_fallback(market_context=market_context, web_results=web_results, question=question)
                else:
                    stale_fix_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    stale_fix_messages.extend(self._trim_history(effective_history))
                    stale_fix_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{runtime_context}"
                                f"{web_context}"
                                f"{profile_context}"
                                f"Question: {question}\n\n"
                                "Give a useful best-effort answer. Use any live context provided above. "
                                "Be explicit about uncertainty but still answer directly."
                            ),
                        }
                    )
                    answer = self.llm.chat(messages=stale_fix_messages, model=model)
            else:
                stale_fix_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                stale_fix_messages.extend(self._trim_history(effective_history))
                stale_fix_messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"{runtime_context}"
                            f"{weather_context}"
                            f"{market_context}"
                            f"{news_context}"
                            f"{web_context}"
                            f"{profile_context}"
                            f"Question: {question}\n\n"
                            "Rewrite your answer using the LIVE WEB RESULTS above as the primary source. "
                            "Do not use training memory for current facts. Do not mention training cutoff dates."
                        ),
                    }
                )
                answer = self.llm.chat(messages=stale_fix_messages, model=model)

        if self._needs_market_context(question):
            forced = self._market_answer_fallback(market_context=market_context, web_results=web_results, question=question)
            if forced and (self._looks_like_market_deflection(answer) or not self._answer_contains_market_snapshot(answer)):
                answer = forced

        answer = self._normalize_response_punctuation(answer)

        if history is None:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
        return answer, hits, web_results, updated_profile

    def ask_stream(self, question, history=None, use_web_search=True,
                   user_timezone=None, user_utc_offset_minutes=None,
                   user_location_label=None, user_profile=None, uploaded_file_context=None,
                   model=None):
        normalized_question = question.strip().lower()
        if normalized_question in ACK_MESSAGES:
            short_reply = "Okay."
            eff_h = history if history is not None else self.history
            up = self._merge_user_profile(user_profile or {}, question, eff_h)
            if history is None:
                self.history.append({"role": "user", "content": question})
                self.history.append({"role": "assistant", "content": short_reply})
            yield ("done", {"answer": short_reply, "hits": [], "web_results": [], "updated_profile": up})
            return
        stream_category = self._classify_query(question)
        stream_is_fresh = self._is_freshness_sensitive(question)
        logger.info("ask_stream_start category=%s freshness_sensitive=%s question=%r", stream_category, stream_is_fresh, question[:100])
        hits = retrieve_context(question, self.chunks, top_k=TOP_K)
        has_strong = bool(hits and hits[0][0] >= MIN_SCORE)
        context = ("\n\n".join(f"[source={src} score={sc:.3f}]\n{txt}" for sc, src, txt in hits) if has_strong else "")
        _effective_web = use_web_search or (ALWAYS_WEB_SEARCH and not self._is_web_search_exempt(question))
        web_results = self._search_live_web(question, max_results=5) if _effective_web else []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        eff_h = history if history is not None else self.history
        updated_profile = self._merge_user_profile(user_profile or {}, question, eff_h)
        profile_ctx = self._profile_context_block(updated_profile)
        messages.extend(self._trim_history(eff_h))
        runtime_ctx = self._runtime_time_context(user_timezone=user_timezone, user_utc_offset_minutes=user_utc_offset_minutes, user_location_label=user_location_label) if (self._needs_runtime_time_context(question) or self._is_current_events_query(question) or self._is_freshness_sensitive(question)) else ""
        weather_ctx = self._weather_context(user_location_label=user_location_label, user_timezone=user_timezone, question=question) if self._needs_weather_context(question) else ""
        market_ctx = self._live_market_context(question=question) if self._needs_market_context(question) else ""
        news_ctx = ""
        if self._needs_news_context(question):
            news_ctx = self._live_news_context()
            tn = self._topic_news_context(question)
            if tn:
                news_ctx += tn
        web_ctx = self._build_web_context_block(web_results)
        # Suppress BM25 note context for freshness-sensitive queries when live web results exist.
        _notes_suppressed = bool(context and self._is_freshness_sensitive(question) and web_results)
        if _notes_suppressed:
            logger.info("notes_suppressed_freshness_sensitive category=%s web_results=%d", stream_category, len(web_results))
        note_blk = ("Context from local notes:\n" + context + "\n\n") if (context and not _notes_suppressed) else ""
        live = bool(web_results) or bool(market_ctx.strip()) or bool(news_ctx.strip()) or bool(weather_ctx.strip())
        anti_r = "" if has_strong else "Instruction: If local notes do not cover this, answer from general knowledge.\n\n"
        anti_s = ("Instruction: Use the live context provided above. Do not reference training cutoff dates.\n\n" if self._is_freshness_sensitive(question) and live else "")
        messages.append({"role": "user", "content": runtime_ctx+weather_ctx+market_ctx+news_ctx+web_ctx+profile_ctx+(uploaded_file_context or "")+note_blk+anti_r+anti_s+"User question:\n"+question})
        streamed = []
        try:
            for token in self.llm.chat_stream(messages=messages, model=model):
                streamed.append(token)
                yield ("token", token)
            answer = "".join(streamed).strip() or "I could not generate a full answer. Please try again."
        except Exception:
            answer = self.llm.chat(messages=messages, model=model)
            yield ("replace", answer)
        if self._looks_like_notes_refusal(answer):
            fbk = [{"role": "system", "content": SYSTEM_PROMPT}]
            fbk.extend(self._trim_history(eff_h))
            fbk.append({"role": "user", "content": runtime_ctx+weather_ctx+market_ctx+news_ctx+web_ctx+profile_ctx+(uploaded_file_context or "")+"Question: "+question+"\n\nGive a direct answer using available live context or general knowledge. If uncertain, state uncertainty briefly, then still provide your best useful answer."})
            answer = self.llm.chat(messages=fbk, model=model)
            yield ("replace", answer)

        # Universal refusal guard: if the model produced a live-retrieval refusal, regenerate once.
        if self._looks_like_refusal_output(answer):
            logger.info("refusal_output_detected_stream question=%r — regenerating", question[:100])
            regen = [{"role": "system", "content": SYSTEM_PROMPT}]
            regen.extend(self._trim_history(eff_h))
            regen.append({"role": "user", "content": runtime_ctx+weather_ctx+market_ctx+news_ctx+web_ctx+profile_ctx+(uploaded_file_context or "")+"Question: "+question+"\n\nGive a direct, concrete, useful answer. Do not say you cannot access real-time or live data. Use live context above if available. Otherwise use general knowledge and note uncertainty briefly."})
            answer = self.llm.chat(messages=regen, model=model)
            yield ("replace", answer)
        if (self._is_current_events_query(question) or self._needs_sports_context(question)) and self._looks_stale_current_answer(answer):
            logger.info("stale_answer_detected_stream question=%r — rerunning with web context", question[:100])
            if not live:
                if self._needs_market_context(question):
                    answer = self._market_answer_fallback(market_context=market_ctx, web_results=web_results, question=question)
                else:
                    sf = [{"role": "system", "content": SYSTEM_PROMPT}]
                    sf.extend(self._trim_history(eff_h))
                    sf.append({"role": "user", "content": runtime_ctx+web_ctx+profile_ctx+"Question: "+question+"\n\nGive a useful best-effort answer. Use any live context provided above. Be explicit about uncertainty but still answer directly."})
                    answer = self.llm.chat(messages=sf, model=model)
                yield ("replace", answer)
            else:
                sf = [{"role": "system", "content": SYSTEM_PROMPT}]
                sf.extend(self._trim_history(eff_h))
                sf.append({"role": "user", "content": runtime_ctx+weather_ctx+market_ctx+news_ctx+web_ctx+profile_ctx+"Question: "+question+"\n\nRewrite your answer using the LIVE WEB RESULTS above as the primary source. Do not use training memory for current facts. Do not mention training cutoff dates."})
                answer = self.llm.chat(messages=sf, model=model)
                yield ("replace", answer)
        if self._needs_market_context(question):
            forced = self._market_answer_fallback(market_context=market_ctx, web_results=web_results, question=question)
            if forced and (self._looks_like_market_deflection(answer) or not self._answer_contains_market_snapshot(answer)):
                answer = forced
                yield ("replace", answer)
        answer = self._normalize_response_punctuation(answer)
        if history is None:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
        yield ("done", {"answer": answer, "hits": hits, "web_results": web_results, "updated_profile": updated_profile, "used_web_search": bool(web_results), "source_count": len(web_results)})

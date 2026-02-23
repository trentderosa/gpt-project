from datetime import datetime, timedelta, timezone
import re
import xml.etree.ElementTree as ET
import time
from zoneinfo import ZoneInfo
import requests
import csv
from io import StringIO

from .config import MIN_SCORE, TOP_K
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
If web results are provided, use them and cite relevant URLs.
If runtime date/time context is provided, use it for questions about today's date, day, or time.
If local notes and web results do not cover a question, use your general model knowledge and be clear when uncertain.
Never answer with only 'I don't know based on my notes'.
For questions about latest/current/live information, do not give outdated cutoff-style answers (for example 'as of 2023').
If live data retrieval fails, still provide a useful best-effort answer and clearly label uncertainty.
Always attempt live/web retrieval before answering.
Use the user profile context to remember the user's name, facts they shared, and mirror their writing style."""

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

CURRENT_QUERY_HINTS = {
    "today",
    "current",
    "latest",
    "right now",
    "this week",
    "this month",
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
}

STALE_ANSWER_HINTS = {
    "as of 2023",
    "as of late 2023",
    "as of my last update",
    "up to my last training",
    "i don't have the latest specific",
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

    def _web_queries(self, question: str) -> list[str]:
        query = question.strip()
        if not query:
            return []
        queries = [query]
        if self._needs_market_context(query):
            queries.extend(
                [
                    "stock market today S&P 500 Dow Nasdaq",
                    "SPY QQQ DIA market update today",
                ]
            )
        if self._is_current_events_query(query):
            queries.append(f"{query} {datetime.now().year}")
            queries.append(f"latest {query}")
        deduped: list[str] = []
        seen = set()
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

    def _search_live_web(self, question: str, max_results: int = 5) -> list[dict]:
        merged: list[dict] = []
        seen_urls: set[str] = set()
        market_query = self._needs_market_context(question)
        for query in self._web_queries(question):
            try:
                items = self.web_search_tool.search(query, max_results=max_results)
            except Exception:
                items = []
            for item in items:
                url = (item.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                if market_query and not self._is_market_result(item):
                    continue
                seen_urls.add(url)
                merged.append(item)
                if len(merged) >= max_results:
                    return merged
        return merged

    def _trim_history(self, history: list[dict], max_messages: int = 8, max_chars: int = 5000) -> list[dict]:
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

    def _looks_stale_current_answer(self, answer: str) -> bool:
        text = (answer or "").lower()
        return any(hint in text for hint in STALE_ANSWER_HINTS)

    def _normalize_response_punctuation(self, text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return "I could not generate a response."

        # Normalize spacing.
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\s+\n", "\n", cleaned).strip()
        cleaned = re.sub(r"([,.!?;:])([A-Za-z])", r"\1 \2", cleaned)
        cleaned = re.sub(r"\bi\b", "I", cleaned)

        # Capitalize sentence starts.
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
        patterns = [
            re.compile(r"\b(?:weather|temp(?:erature)?)\s+(?:in|for|at)\s+([A-Za-z0-9.,\-\s]{2,80})\??$", re.IGNORECASE),
            re.compile(r"\bin\s+([A-Za-z0-9.,\-\s]{2,80})\s*$", re.IGNORECASE),
        ]
        for pattern in patterns:
            match = pattern.search(q)
            if match:
                location = match.group(1).strip(" .,!?:;")
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
            return (
                "Live weather context:\n"
                "- Could not determine weather location. Ask user to enable location access or provide city and state/country.\n\n"
            )
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
            return "Live weather context:\n- Weather lookup failed for this request.\n\n"

    def _extract_symbols_from_question(self, question: str) -> list[str]:
        # Pick uppercase ticker-style tokens, $-prefixed symbols, and common lowercase ticker mentions.
        dollar_tokens = [m.upper() for m in re.findall(r"\$([A-Za-z]{1,5})\b", question)]
        candidates = re.findall(r"\b[A-Z]{1,5}\b", question)
        lowered = question.lower()
        common = [
            "spy",
            "qqq",
            "dia",
            "iwm",
            "aapl",
            "msft",
            "nvda",
            "tsla",
            "amzn",
            "meta",
            "googl",
            "gspc",
            "dji",
            "ixic",
        ]
        common_hits = [sym.upper() for sym in common if re.search(rf"\b{re.escape(sym)}\b", lowered)]
        deny = {"I", "A", "AN", "THE", "AND", "OR", "IT", "WE", "YOU"}
        symbols = [c for c in dollar_tokens + candidates + common_hits if c not in deny]
        deduped: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            if symbol in seen:
                continue
            seen.add(symbol)
            deduped.append(symbol)
        symbols = deduped
        return symbols[:8]

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
            return "Live market context:\n- Market lookup failed for this request.\n\n"
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

    def _live_news_context(self) -> str:
        rows = self.live_store.get_latest("news", limit=5)
        if not rows:
            return "Live news context:\n- No cached news snapshots are available.\n\n"
        lines = ["Live news context (latest headlines):"]
        for row in rows[:5]:
            payload = row.get("payload", {})
            items = payload.get("items", [])
            for item in items[:4]:
                title = (item.get("title") or "").strip() or "Untitled"
                link = (item.get("link") or "").strip()
                pub_date = (item.get("pub_date") or "").strip()
                lines.append(f"- {title} ({pub_date}) {link}")
        return "\n".join(lines) + "\n\n"

    def _extract_market_lines(self, market_context: str) -> list[str]:
        lines = []
        for line in (market_context or "").splitlines():
            clean = line.strip()
            if clean.startswith("- ") and ("price=" in clean or "close=" in clean):
                lines.append(clean)
        return lines[:5]

    def _answer_contains_market_snapshot(self, answer: str) -> bool:
        text = (answer or "").lower()
        return "latest market snapshot" in text or bool(re.search(r"\b[a-z0-9\^]{1,6}\s*:\s*(price|close)\s*=", text))

    def _market_answer_fallback(self, market_context: str, web_results: list[dict]) -> str:
        market_lines = self._extract_market_lines(market_context)
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

        hits = retrieve_context(question, self.chunks, top_k=TOP_K)
        has_strong_note_context = bool(hits and hits[0][0] >= MIN_SCORE)
        if has_strong_note_context:
            context = "\n\n".join(
                f"[source={source} score={score:.3f}]\n{text}" for score, source, text in hits
            )
        else:
            context = ""

        web_results: list[dict] = []
        if use_web_search:
            web_results = self._search_live_web(question, max_results=5)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        effective_history = history if history is not None else self.history
        updated_profile = self._merge_user_profile(user_profile or {}, question, effective_history)
        profile_context = self._profile_context_block(updated_profile)
        messages.extend(self._trim_history(effective_history))
        runtime_context = ""
        if self._needs_runtime_time_context(question) or self._is_current_events_query(question):
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
        web_context = (
            "\n\nWeb results:\n"
            + "\n".join(
                f"- {item.get('title', 'Untitled')}: {item.get('snippet', '')} ({item.get('url', '')})"
                for item in web_results
            )
            if web_results
            else ""
        )
        note_context_block = ""
        if context:
            note_context_block = f"Context from local notes:\n{context}\n\n"
        messages.append(
            {
                "role": "user",
                "content": (
                    f"{runtime_context}"
                    f"{weather_context}"
                    f"{market_context}"
                    f"{news_context}"
                    f"{profile_context}"
                    f"{uploaded_file_context or ''}"
                    f"{note_context_block}"
                    f"User question:\n{question}{web_context}"
                ),
            }
        )

        answer = self.llm.chat(messages=messages)
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
                        f"{profile_context}"
                        f"{uploaded_file_context or ''}"
                        f"Question: {question}\n\n"
                        "Give a direct answer using general knowledge. "
                        "If uncertain, state uncertainty briefly, then still provide your best useful answer."
                    ),
                }
            )
            answer = self.llm.chat(messages=fallback_messages)

        if self._is_current_events_query(question) and self._looks_stale_current_answer(answer):
            has_live_signal = bool(web_results) or bool(market_context.strip()) or bool(news_context.strip()) or bool(
                weather_context.strip()
            )
            if not has_live_signal:
                if self._needs_market_context(question):
                    answer = self._market_answer_fallback(market_context=market_context, web_results=web_results)
                else:
                    stale_fix_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    stale_fix_messages.extend(self._trim_history(effective_history))
                    stale_fix_messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"{runtime_context}"
                                f"{profile_context}"
                                f"Question: {question}\n\n"
                                "Give a useful best-effort answer without claiming live retrieval is unavailable. "
                                "Be explicit about uncertainty, but still answer directly."
                            ),
                        }
                    )
                    answer = self.llm.chat(messages=stale_fix_messages)
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
                            f"{profile_context}"
                            f"Question: {question}\n\n"
                            "Rewrite your answer using available live context only. "
                            "Do not mention training cutoff dates."
                        ),
                    }
                )
                answer = self.llm.chat(messages=stale_fix_messages)

        if self._needs_market_context(question):
            forced = self._market_answer_fallback(market_context=market_context, web_results=web_results)
            if forced and (self._looks_like_market_deflection(answer) or not self._answer_contains_market_snapshot(answer)):
                answer = forced

        answer = self._normalize_response_punctuation(answer)

        if history is None:
            self.history.append({"role": "user", "content": question})
            self.history.append({"role": "assistant", "content": answer})
        return answer, hits, web_results, updated_profile

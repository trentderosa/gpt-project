import logging
import os
from typing import Iterable

import requests


logger = logging.getLogger("cortex.search")


class WebSearchTool:
    provider_name = "unknown"
    provider_kind = "search"

    def is_configured(self) -> bool:
        return True

    def search(
        self,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        raise NotImplementedError


class DisabledWebSearchTool(WebSearchTool):
    provider_name = "disabled"

    def is_configured(self) -> bool:
        return False

    def search(
        self,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        return []


class DuckDuckGoSearchTool(WebSearchTool):
    provider_name = "ddg"

    def _query_with_domains(self, query: str, preferred_domains: list[str] | None) -> str:
        if not preferred_domains:
            return query
        domains = [d.strip() for d in preferred_domains if d and d.strip() and not d.startswith(".")]
        if not domains:
            return query
        if len(domains) == 1:
            return f"{query} site:{domains[0]}"
        filters = " OR ".join(f"site:{domain}" for domain in domains[:4])
        return f"{query} ({filters})"

    def search(
        self,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        try:
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS
        except Exception:
            return []

        results: list[dict] = []
        request_query = self._query_with_domains(query, preferred_domains)
        try:
            with DDGS() as ddgs:
                timelimit = "m" if fresh else None
                for item in ddgs.text(request_query, max_results=max_results, timelimit=timelimit):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "snippet": item.get("body", ""),
                            "url": item.get("href", ""),
                            "date": item.get("published", ""),
                            "provider": self.provider_name,
                        }
                    )
        except Exception:
            return []
        return results


class WikipediaSearchTool(WebSearchTool):
    provider_name = "wikipedia"
    provider_kind = "reference"

    def _query_with_domains(self, query: str, preferred_domains: list[str] | None) -> str:
        return query

    def search(
        self,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        q = (query or "").strip()
        if not q:
            return []
        try:
            resp = requests.get(
                "https://en.wikipedia.org/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": q,
                    "srlimit": max_results,
                    "format": "json",
                },
                timeout=8,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = ((payload.get("query") or {}).get("search") or [])
            results: list[dict] = []
            for item in items[:max_results]:
                title = (item.get("title") or "").strip()
                snippet = (item.get("snippet") or "").replace("<span class=\"searchmatch\">", "").replace("</span>", "")
                url_title = title.replace(" ", "_")
                results.append(
                    {
                        "title": title,
                        "snippet": snippet,
                        "url": f"https://en.wikipedia.org/wiki/{url_title}",
                        "provider": self.provider_name,
                    }
                )
            return results
        except Exception:
            return []


class BraveSearchTool(WebSearchTool):
    provider_name = "brave"

    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _query_with_domains(self, query: str, preferred_domains: list[str] | None) -> str:
        if not preferred_domains:
            return query
        domains = [d.strip() for d in preferred_domains if d and d.strip() and not d.startswith(".")]
        if not domains:
            return query
        if len(domains) == 1:
            return f"{query} site:{domains[0]}"
        filters = " OR ".join(f"site:{domain}" for domain in domains[:4])
        return f"{query} ({filters})"

    def search(
        self,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        if not self.api_key:
            return []
        try:
            params: dict = {
                "q": self._query_with_domains(query, preferred_domains),
                "count": max_results,
                "country": "us",
            }
            if fresh:
                params["freshness"] = "pm"
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                params=params,
                timeout=8,
            )
            resp.raise_for_status()
            payload = resp.json()
            items = ((payload.get("web") or {}).get("results") or [])
            results: list[dict] = []
            for item in items[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("description", ""),
                        "url": item.get("url", ""),
                        "date": item.get("age", "") or item.get("page_age", ""),
                        "provider": self.provider_name,
                    }
                )
            return results
        except Exception:
            return []


class TavilySearchTool(WebSearchTool):
    provider_name = "tavily"

    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _topic_for_category(self, category: str | None) -> str:
        if category in {"news", "current_events", "leadership", "sports", "sports_event"}:
            return "news"
        if category == "finance":
            return "finance"
        return "general"

    def search(
        self,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        if not self.api_key:
            return []
        payload: dict = {
            "query": query,
            "max_results": max_results,
            "search_depth": "advanced" if fresh or preferred_domains else "basic",
            "topic": self._topic_for_category(category),
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }
        if fresh:
            payload["time_range"] = "month"
        if preferred_domains:
            payload["include_domains"] = preferred_domains[:20]
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=12,
            )
            resp.raise_for_status()
            body = resp.json()
            items = body.get("results") or []
            results: list[dict] = []
            for item in items[:max_results]:
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("content", ""),
                        "url": item.get("url", ""),
                        "date": item.get("published_date", ""),
                        "score": item.get("score", 0),
                        "provider": self.provider_name,
                    }
                )
            return results
        except Exception:
            return []


class CompositeWebSearchTool(WebSearchTool):
    provider_name = "composite"

    def __init__(
        self,
        providers: list[WebSearchTool],
        provider_priority: list[str] | None = None,
        ddg_fallback_only: bool = True,
    ):
        self.providers = providers
        self.provider_priority = [name.strip().lower() for name in (provider_priority or []) if name and name.strip()]
        self.ddg_fallback_only = ddg_fallback_only
        self._provider_map = {provider.provider_name: provider for provider in providers}
        self._last_trace: list[dict] = []

    def is_configured(self) -> bool:
        return any(provider.is_configured() for provider in self.providers)

    def available_providers(self) -> list[str]:
        return [provider.provider_name for provider in self.providers if provider.is_configured()]

    def get_provider(self, provider_name: str) -> WebSearchTool | None:
        provider = self._provider_map.get((provider_name or "").strip().lower())
        if provider and provider.is_configured():
            return provider
        return None

    def get_last_trace(self) -> list[dict]:
        return [dict(item) for item in self._last_trace]

    def _ordered_provider_names(self) -> list[str]:
        ordered: list[str] = []
        for name in self.provider_priority:
            if self.get_provider(name) and name not in ordered:
                ordered.append(name)
        for provider in self.providers:
            if provider.is_configured() and provider.provider_name not in ordered:
                ordered.append(provider.provider_name)
        return ordered

    def provider_sequence(self, category: str | None = None, fresh: bool = False) -> list[str]:
        ordered = self._ordered_provider_names()
        primaries = [name for name in ordered if name not in {"ddg", "wikipedia"}]
        fallbacks = [name for name in ordered if name in {"ddg", "wikipedia"}]
        sequence = list(primaries)
        if self.ddg_fallback_only and "ddg" in fallbacks:
            fallbacks = [name for name in fallbacks if name != "ddg"] + ["ddg"]
        if fresh or category in {"weather", "finance", "sports", "sports_event", "leadership", "news", "current_events"}:
            return sequence + [name for name in fallbacks if name == "ddg"]
        return sequence + fallbacks

    def search_with_provider(
        self,
        provider_name: str,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        provider = self.get_provider(provider_name)
        if not provider:
            logger.info("web_search_provider_unavailable provider=%s", provider_name)
            return []
        try:
            items = provider.search(
                query=query,
                max_results=max_results,
                fresh=fresh,
                category=category,
                preferred_domains=preferred_domains,
            )
        except Exception as exc:
            logger.warning(
                "web_search_provider_error provider=%s query=%r fresh=%s error=%s",
                provider_name,
                query[:100],
                fresh,
                exc,
            )
            items = []
        self._last_trace.append(
            {
                "provider": provider_name,
                "query": query,
                "fresh": fresh,
                "category": category,
                "preferred_domains": list(preferred_domains or []),
                "result_count": len(items),
            }
        )
        return items

    def search(
        self,
        query: str,
        max_results: int = 3,
        fresh: bool = False,
        category: str | None = None,
        preferred_domains: list[str] | None = None,
    ) -> list[dict]:
        self._last_trace = []
        for provider_name in self.provider_sequence(category=category, fresh=fresh):
            items = self.search_with_provider(
                provider_name=provider_name,
                query=query,
                max_results=max_results,
                fresh=fresh,
                category=category,
                preferred_domains=preferred_domains,
            )
            if items:
                return items[:max_results]
        return []


def _parse_priority(value: str | None) -> list[str]:
    raw = (value or "").strip()
    if not raw:
        return []
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def build_search_tool(
    web_search_provider: str | None = None,
    provider_priority: Iterable[str] | None = None,
    brave_api_key: str | None = None,
    tavily_api_key: str | None = None,
) -> WebSearchTool:
    provider_mode = (web_search_provider or os.getenv("WEB_SEARCH_PROVIDER", "auto")).strip().lower()
    brave_key = (brave_api_key if brave_api_key is not None else os.getenv("BRAVE_SEARCH_API_KEY", "")).strip()
    tavily_key = (tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY", "")).strip()

    if provider_mode in {"disabled", "off", "none"}:
        return DisabledWebSearchTool()

    if provider_mode == "duckduckgo":
        return CompositeWebSearchTool(providers=[DuckDuckGoSearchTool(), WikipediaSearchTool()], provider_priority=["ddg", "wikipedia"], ddg_fallback_only=False)

    resolved_priority = list(provider_priority or _parse_priority(os.getenv("WEB_SEARCH_PROVIDER_PRIORITY")))
    if not resolved_priority:
        resolved_priority = ["brave", "tavily", "ddg", "wikipedia"]

    providers: list[WebSearchTool] = []
    if brave_key:
        providers.append(BraveSearchTool(brave_key))
    if tavily_key:
        providers.append(TavilySearchTool(tavily_key))
    providers.append(DuckDuckGoSearchTool())
    providers.append(WikipediaSearchTool())

    tool = CompositeWebSearchTool(
        providers=providers,
        provider_priority=resolved_priority,
        ddg_fallback_only=True,
    )
    if provider_mode in {"auto", "multi", "brave", "tavily"}:
        logger.info(
            "web_search_tool_configured mode=%s providers=%s priority=%s brave=%s tavily=%s",
            provider_mode,
            tool.available_providers(),
            resolved_priority,
            bool(brave_key),
            bool(tavily_key),
        )
        if not brave_key and not tavily_key:
            logger.warning("web_search_primary_providers_missing mode=%s providers=%s", provider_mode, tool.available_providers())
        return tool
    return DisabledWebSearchTool()

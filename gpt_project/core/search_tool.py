import requests


class WebSearchTool:
    def search(self, query: str, max_results: int = 3) -> list[dict]:
        raise NotImplementedError


class DisabledWebSearchTool(WebSearchTool):
    def search(self, query: str, max_results: int = 3) -> list[dict]:
        return []


class DuckDuckGoSearchTool(WebSearchTool):
    def search(self, query: str, max_results: int = 3) -> list[dict]:
        try:
            from duckduckgo_search import DDGS
        except Exception:
            return []

        results: list[dict] = []
        try:
            with DDGS() as ddgs:
                for item in ddgs.text(query, max_results=max_results):
                    results.append(
                        {
                            "title": item.get("title", ""),
                            "snippet": item.get("body", ""),
                            "url": item.get("href", ""),
                            "date": item.get("published", ""),
                        }
                    )
        except Exception:
            return []
        return results


class WikipediaSearchTool(WebSearchTool):
    def search(self, query: str, max_results: int = 3) -> list[dict]:
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
                    }
                )
            return results
        except Exception:
            return []


class BraveSearchTool(WebSearchTool):
    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        if not self.api_key:
            return []
        try:
            resp = requests.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                params={"q": query, "count": max_results},
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
                    }
                )
            return results
        except Exception:
            return []


class CompositeWebSearchTool(WebSearchTool):
    def __init__(self, providers: list[WebSearchTool]):
        self.providers = providers

    def search(self, query: str, max_results: int = 3) -> list[dict]:
        merged: list[dict] = []
        seen: set[str] = set()
        for provider in self.providers:
            try:
                items = provider.search(query, max_results=max_results)
            except Exception:
                items = []
            for item in items:
                url = (item.get("url") or "").strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                merged.append(item)
                if len(merged) >= max_results:
                    return merged
        return merged

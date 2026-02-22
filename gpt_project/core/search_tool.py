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
                        }
                    )
        except Exception:
            return []
        return results


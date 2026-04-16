from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BASE_DIR / "app.py" / "knowledge"
WEB_DIR = BASE_DIR / "gpt_project" / "web"

DEFAULT_MODEL = "gpt-4o-mini"
TOP_K = 4
CHUNK_SIZE = 700
MIN_SCORE = 1.5

WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "multi").strip().lower()
WEB_SEARCH_PROVIDER_PRIORITY = os.getenv("WEB_SEARCH_PROVIDER_PRIORITY", "brave,tavily,ddg,wikipedia").strip()
BRAVE_SEARCH_API_KEY = (os.getenv("BRAVE_SEARCH_API_KEY") or "").strip()
TAVILY_API_KEY = (os.getenv("TAVILY_API_KEY") or "").strip()

# When True, the server forces web search on all non-exempt queries regardless of
# the client's use_web_search flag.  Set ALWAYS_WEB_SEARCH=false to disable.
ALWAYS_WEB_SEARCH: bool = os.getenv("ALWAYS_WEB_SEARCH", "true").strip().lower() not in ("0", "false", "no", "off")

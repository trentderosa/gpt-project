from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = BASE_DIR / "app.py" / "knowledge"
WEB_DIR = BASE_DIR / "gpt_project" / "web"

DEFAULT_MODEL = "gpt-4o-mini"
TOP_K = 4
CHUNK_SIZE = 700
MIN_SCORE = 0.15

WEB_SEARCH_PROVIDER = os.getenv("WEB_SEARCH_PROVIDER", "duckduckgo").strip().lower()

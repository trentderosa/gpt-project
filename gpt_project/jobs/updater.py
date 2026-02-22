import logging
import os
import xml.etree.ElementTree as ET

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

from ..core.live_data_store import LiveDataStore


logger = logging.getLogger("trent_gpt_worker")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

DEFAULT_NEWS_FEEDS = [
    "https://feeds.reuters.com/reuters/topNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml",
]


def _get_news_feeds() -> list[str]:
    raw = (os.getenv("NEWS_FEED_URLS") or "").strip()
    if not raw:
        return DEFAULT_NEWS_FEEDS
    return [item.strip() for item in raw.split(",") if item.strip()]


def _get_stock_symbols() -> list[str]:
    raw = (os.getenv("STOCK_SYMBOLS") or "AAPL,MSFT,SPY,TSLA,NVDA").strip()
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def fetch_stock_quotes(symbols: list[str]) -> list[dict]:
    # Stooq public CSV endpoint; good for free snapshots.
    symbol_param = ",".join(f"{symbol.lower()}.us" for symbol in symbols)
    url = f"https://stooq.com/q/l/?s={symbol_param}&f=sd2t2ohlcv&h&e=csv"
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    lines = [line.strip() for line in response.text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return []
    headers = [h.strip().lower() for h in lines[0].split(",")]
    rows: list[dict] = []
    for line in lines[1:]:
        values = [v.strip() for v in line.split(",")]
        rows.append(dict(zip(headers, values)))
    return rows


def fetch_feed_items(feed_url: str, max_items: int = 12) -> list[dict]:
    response = requests.get(feed_url, timeout=20)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    items = []
    for item in root.findall(".//item")[:max_items]:
        items.append(
            {
                "title": (item.findtext("title") or "").strip(),
                "link": (item.findtext("link") or "").strip(),
                "pub_date": (item.findtext("pubDate") or "").strip(),
                "description": (item.findtext("description") or "").strip(),
            }
        )
    return items


def run_once(store: LiveDataStore) -> None:
    logger.info("live_update_cycle_start")

    symbols = _get_stock_symbols()
    if symbols:
        try:
            rows = fetch_stock_quotes(symbols)
            for row in rows:
                key = row.get("symbol", "unknown")
                store.upsert_snapshot("stock", key, row)
            logger.info("stocks_updated count=%s", len(rows))
        except Exception:
            logger.exception("stocks_update_failed")

    feeds = _get_news_feeds()
    for feed in feeds:
        try:
            items = fetch_feed_items(feed, max_items=12)
            store.upsert_snapshot("news", feed, {"items": items})
            logger.info("news_updated feed=%s count=%s", feed, len(items))
        except Exception:
            logger.exception("news_update_failed feed=%s", feed)

    logger.info("live_update_cycle_complete")


def run_worker() -> None:
    enabled = os.getenv("LIVE_UPDATE_ENABLED", "true").strip().lower() == "true"
    if not enabled:
        logger.info("worker_disabled LIVE_UPDATE_ENABLED=false")
        return

    interval = int(os.getenv("LIVE_UPDATE_INTERVAL_SECONDS", "300"))
    store = LiveDataStore()

    scheduler = BlockingScheduler()
    scheduler.add_job(run_once, "interval", seconds=interval, args=[store], max_instances=1)

    # Run one cycle immediately at startup.
    run_once(store)
    logger.info("worker_started interval_seconds=%s", interval)
    scheduler.start()


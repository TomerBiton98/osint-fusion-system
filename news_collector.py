"""
Unified OSINT News Collector
RSS (Realtime) + NYT Archive (Historical baseline)
"""

import feedparser
import logging
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    title: str
    snippet: str
    url: str
    publisher: str
    published_at: datetime
    credibility: float = 0.7

    @property
    def date(self):
        return self.published_at

class RSSCollector:
    FEEDS = {
        "Reuters": "https://www.reuters.com/rssFeed/worldNews",
        "BBC": "https://feeds.bbci.co.uk/news/world/rss.xml",
        "AlJazeera": "https://www.aljazeera.com/xml/rss/all.xml",

        "AlJazeera Middle East": "https://www.aljazeera.com/xml/rss/all.xml",
        "Middle East Eye": "https://www.middleeasteye.net/rss",
        "Times of Israel": "https://www.timesofisrael.com/feed/",
        "Haaretz": "https://www.haaretz.com/rss",

        "AP News": "https://apnews.com/rss",
        "Politico": "https://www.politico.com/rss/politics08.xml",
        "Defense News": "https://www.defensenews.com/arc/outboundfeeds/rss/",

        "ReliefWeb": "https://reliefweb.int/rss.xml",
        "UN News": "https://news.un.org/feed/subscribe/en/news/all/rss.xml",

    }

    def collect(self, keywords: List[str], max_articles=30) -> List[NewsArticle]:
        results = []
        keywords_l = [k.lower() for k in keywords]

        for publisher, url in self.FEEDS.items():
            feed = feedparser.parse(url)

            for entry in feed.entries:
                text = f"{entry.title} {entry.get('summary','')}".lower()
                matched = [k for k in keywords_l if k in text]
                if keywords_l and not matched:
                    continue

                logger.debug(f"[MATCH] {publisher} → {matched}")

                published = (
                    datetime(*entry.published_parsed[:6])
                    if hasattr(entry, "published_parsed")
                    else datetime.utcnow()
                )

                results.append(
                    NewsArticle(
                        title=entry.title,
                        snippet=entry.get("summary", "")[:500],
                        url=entry.link,
                        publisher=publisher,
                        published_at=published,
                        credibility=0.85 if publisher == "Reuters" else 0.75,
                    )
                )

                if len(results) >= max_articles:
                    return results

        return results

class OSINTNewsCollector:
    def __init__(self, nyt_archive: Optional[object] = None):
        self.rss = RSSCollector()
        self.nyt_archive = nyt_archive

    def collect(self, keywords: List[str], max_articles=30) -> List[NewsArticle]:
        articles = self.rss.collect(keywords, max_articles)

        if self.nyt_archive:
            try:
                baseline = self.nyt_archive.baseline_signal(keywords)
                logger.info(f"NYT baseline signal (historical): {baseline}")
            except Exception as e:
                logger.warning(f"NYT Archive failed: {e}")

        return articles


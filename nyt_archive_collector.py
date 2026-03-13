import requests
from typing import List
from datetime import datetime

class NYTArchiveCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_month(self, year: int, month: int) -> List[str]:
        url = f"https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"
        r = requests.get(url, params={"api-key": self.api_key}, timeout=30)
        r.raise_for_status()
        docs = r.json()["response"]["docs"]
        return [d["headline"]["main"] for d in docs]

    def baseline_signal(self, keywords: List[str], years_back=5) -> int:
        now = datetime.utcnow()
        count = 0

        for y in range(now.year - years_back, now.year):
            headlines = self.fetch_month(y, now.month)
            for h in headlines:
                if any(k.lower() in h.lower() for k in keywords):
                    count += 1
        return count

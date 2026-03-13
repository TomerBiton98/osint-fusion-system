from typing import List, Dict

def compute_confidence(articles: List, imagery: Dict) -> float:
    if not articles:
        return 0.0

    news_strength = min(len(articles) / 5, 1.0)

    sat_strength = 0.0
    if imagery.get("s1") and imagery["s1"].get("status") == "OK":
        sat_strength += 0.5
    if imagery.get("s2") and imagery["s2"].get("status") == "OK":
        sat_strength += 0.5

    return round(news_strength * 0.6 + sat_strength * 0.4, 2)


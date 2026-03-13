import os
import json
import logging
import traceback
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from dataclasses import asdict

from dotenv import load_dotenv

from nyt_archive_collector import NYTArchiveCollector
from news_collector import OSINTNewsCollector, NewsArticle
from ai_forecaster import AIForecaster
from confidence import compute_confidence

from osint_fusion_system import (
    ImageryRequest,
    SatelliteMode,
    CopernicusClient,
    RasterValidator,
    ImageryRenderer,
    EventExtractor,
)

logger = logging.getLogger(__name__)

def _json_safe(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value

class OSINTFusionPipeline:
    def __init__(self, output_dir="./outputs", env_path=".env"):
        load_dotenv(env_path)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            "sh_client_id": os.getenv("SH_CLIENT_ID"),
            "sh_client_secret": os.getenv("SH_CLIENT_SECRET"),
            "nyt_api_key": os.getenv("NYT_API_KEY"),
        }

        self.nyt_archive = (
            NYTArchiveCollector(self.config["nyt_api_key"])
            if self.config["nyt_api_key"]
            else None
        )

        self.copernicus_client = (
            CopernicusClient(
                self.config["sh_client_id"],
                self.config["sh_client_secret"],
            )
            if self.config["sh_client_id"] and self.config["sh_client_secret"]
            else None
        )

        self.news_collector = OSINTNewsCollector(self.nyt_archive)
        self.ai_forecaster = AIForecaster()

    def run_analysis(self, request: ImageryRequest) -> Dict:
        job_id = f"job_{request.aoi_name.replace(' ', '_')}_{datetime.utcnow():%Y%m%d_%H%M%S}"
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        results = {
            "job_id": job_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "request": _json_safe(asdict(request)),
            "status": "running",
            "imagery": {},
            "news": {},
            "forecast": {},
            "errors": [],
        }

        try:
            imagery = self._acquire_imagery(request, job_dir, results)
            articles = self._collect_news(request, job_dir, results)
            claims = self._extract_claims(articles, results)
            self._generate_forecast(request, articles, claims, imagery, results)
            self._generate_report(job_dir, results)

            results["status"] = "completed"

        except Exception as e:
            logger.error(traceback.format_exc())
            results["status"] = "failed"
            results["errors"].append(str(e))

        self._save_results(job_dir, results)
        return results

    def _acquire_imagery(self, request, job_dir, results):
        imagery = {}

        if not self.copernicus_client:
            return imagery

        if request.satellite_mode in [SatelliteMode.AUTO, SatelliteMode.S1]:
            s1 = job_dir / "sentinel1_raw.tif"
            status, _ = self.copernicus_client.acquire_sentinel1(
                request.bbox, request.time_range, str(s1)
            )
            imagery["s1"] = {"status": status.value, "path": str(s1)}

        if request.satellite_mode in [SatelliteMode.AUTO, SatelliteMode.S2]:
            s2 = job_dir / "sentinel2_raw.tif"
            status, _ = self.copernicus_client.acquire_sentinel2(
                request.bbox, request.time_range, str(s2)
            )
            imagery["s2"] = {"status": status.value, "path": str(s2)}

        for k, entry in imagery.items():
            if entry["path"]:
                val = RasterValidator.validate(entry["path"])
                if val.is_valid:
                    png = job_dir / f"{k}_viz.png"
                    renderer = (
                        ImageryRenderer.render_sar
                        if k == "s1"
                        else ImageryRenderer.render_optical
                    )
                    renderer(entry["path"], str(png))
                    entry["visualization"] = str(png)

        results["imagery"]["acquisitions"] = imagery
        return imagery

    def _collect_news(self, request, job_dir, results):
        articles = self.news_collector.collect(
            keywords=request.news_keywords or ["news"], max_articles=30
        )

        news_path = job_dir / "news.json"
        with open(news_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "title": a.title,
                        "snippet": a.snippet,
                        "url": a.url,
                        "publisher": a.publisher,
                        "published_at": a.published_at.isoformat(),
                        "credibility": a.credibility,
                    }
                    for a in articles
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )

        results["news"] = {
            "article_count": len(articles),
            "articles": [
                {
                    "title": a.title,
                    "publisher": a.publisher,
                    "url": a.url,
                    "published_at": a.published_at.isoformat(),
                }
                for a in articles
            ],
        }

        return articles

    def _extract_claims(self, articles, results):
        extractor = EventExtractor()
        claims = extractor.extract_claims(articles) if articles else []
        results["news"]["claim_count"] = len(claims)
        return claims

    def _generate_forecast(self, request, articles, claims, imagery, results):
        confidence = compute_confidence(articles, imagery)
        texts = [f"{a.title}. {a.snippet}" for a in articles[:10]]
        event_classes = self.ai_forecaster.classify_event(" ".join(texts))
        urgency, trend = self.ai_forecaster.analyze_urgency(texts)

        results["forecast"] = {
            "event_classes": event_classes,
            "urgency_score": urgency,
            "trend": trend,
            "possible_outcomes": [
                {"scenario": "De-escalation", "probability": round(0.6 - urgency * 0.2, 2)},
                {"scenario": "Sustained tension", "probability": round(0.3 + urgency * 0.1, 2)},
                {"scenario": "Escalation", "probability": round(0.1 + urgency * 0.1, 2)},
            ],
            "confidence": confidence,
        }

    def _generate_report(self, job_dir, results):
        md = job_dir / "report.md"
        with open(md, "w", encoding="utf-8") as f:
            f.write("# OSINT Fusion Report\n\n")
            f.write(json.dumps(results["forecast"], indent=2))
        results["report_md"] = str(md)

    def _save_results(self, job_dir, results):
        with open(job_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(_json_safe(results), f, indent=2)


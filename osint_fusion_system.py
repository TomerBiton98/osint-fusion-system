from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")

from urllib.parse import quote
import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import rasterio
from rasterio.warp import transform_bounds
import requests
from pathlib import Path
import hashlib
from PIL import Image
import io

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SatelliteMode(Enum):
    S1 = "sentinel-1"
    S2 = "sentinel-2"
    AUTO = "auto"

class OrbitDirection(Enum):
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    ANY = "ANY"

class RiskProfile(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class DataStatus(Enum):
    OK = "OK"
    NO_DATA = "NO_DATA"
    INVALID_PRODUCT = "INVALID_PRODUCT"
    AUTH_FAILED = "AUTH_FAILED"
    API_ERROR = "API_ERROR"

@dataclass
class ImageryRequest:
    """Input parameters for imagery acquisition"""
    bbox: List[float]  

    time_range: Dict[str, str]  

    aoi_name: Optional[str] = None
    satellite_mode: SatelliteMode = SatelliteMode.AUTO
    orbit_direction: OrbitDirection = OrbitDirection.ANY
    news_keywords: Optional[List[str]] = None
    language_filter: Optional[List[str]] = None
    risk_profile: RiskProfile = RiskProfile.MEDIUM
    output_size: Tuple[int, int] = (2500, 2500)

@dataclass
class SatelliteAcquisition:
    """Metadata for a satellite acquisition"""
    sensor: str
    acquisition_time: str
    orbit_direction: str
    polarization: Optional[str] = None
    cloud_coverage: Optional[float] = None
    product_id: str = ""

@dataclass
class ValidationResult:
    """Results from raster validation"""
    is_valid: bool
    status: DataStatus
    reason: str
    file_size_mb: float
    shape: Optional[Tuple] = None
    nodata_ratio: float = 0.0
    stats: Optional[Dict[str, float]] = None

@dataclass
class NewsArticle:
    """Structured news article"""
    title: str
    publisher: str
    url: str
    date: str
    snippet: str
    full_text: Optional[str] = None
    relevance_score: float = 0.0
    credibility_score: float = 0.5
    location_mentions: Optional[List[str]] = None

@dataclass
class EventClaim:
    """Extracted event claim from news"""
    what: str
    where: str
    when: str
    who: Optional[str] = None
    casualties: Optional[str] = None
    damage: Optional[str] = None
    uncertainty: str = "medium"
    sources: List[str] = None

@dataclass
class Prediction:
    """AI-generated prediction"""
    hypothesis: str
    confidence: float
    signals: List[str]
    what_to_verify_next: List[str]

@dataclass
class AIForecast:
    """Complete AI forecast output"""
    status: str
    forecast_horizon_hours: int
    predictions: List[Prediction]
    caveats: List[str]
    model_versions: Dict[str, str]

class CopernicusClient:
    """Client for Copernicus Dataspace / Sentinel Hub API"""

    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.token = None
        self.token_expires = None
        self.base_url = "https://sh.dataspace.copernicus.eu"

    def authenticate(self) -> bool:
        """Authenticate and get OAuth token"""
        try:
            url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
            data = {
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret
            }
            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()

            token_data = response.json()
            self.token = token_data["access_token"]
            self.token_expires = datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600))

            logger.info("Successfully authenticated with Copernicus")
            return True

        except Exception as e:
            logger.error(f"Authentication failed: {e}")
            return False

    def _ensure_token(self):
        """Ensure we have a valid token"""
        if not self.token or datetime.now() >= self.token_expires:
            if not self.authenticate():
                raise Exception("Failed to authenticate")

    def check_catalogue(self, bbox: List[float], time_range: Dict[str, str], 
                       collection: str) -> Tuple[bool, List[SatelliteAcquisition]]:
        """
        Query catalogue to check data availability
        Returns: (data_exists, list_of_acquisitions)
        """
        self._ensure_token()

        try:

            url = f"{self.base_url}/api/v1/catalog/1.0.0/search"

            payload = {
                "bbox": bbox,
                "datetime": f"{time_range['from']}/{time_range['to']}",
                "collections": [collection],
                "limit": 100
            }

            headers = {"Authorization": f"Bearer {self.token}"}
            response = requests.post(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                features = response.json().get("features", [])

                acquisitions = []
                for feature in features:
                    props = feature.get("properties", {})
                    acq = SatelliteAcquisition(
                        sensor=collection,
                        acquisition_time=props.get("datetime", ""),
                        orbit_direction=props.get("sat:orbit_state", "UNKNOWN"),
                        polarization=props.get("sar:polarizations", None),
                        cloud_coverage=props.get("eo:cloud_cover", None),
                        product_id=feature.get("id", "")
                    )
                    acquisitions.append(acq)

                return len(acquisitions) > 0, acquisitions
            else:
                logger.warning(f"Catalogue query returned status {response.status_code}")
                return False, []

        except Exception as e:
            logger.error(f"Catalogue check failed: {e}")
            return False, []

    def acquire_sentinel1(self, bbox: List[float], time_range: Dict[str, str],
                         output_path: str, width: int = 512, height: int = 512,
                         polarization: str = "VV") -> Tuple[DataStatus, str]:
        """Acquire Sentinel-1 SAR imagery"""
        self._ensure_token()

        try:

            url = f"{self.base_url}/api/v1/process"

            evalscript = f"""
//VERSION=3
function setup() {{
  return {{
    input: [{{
      bands: ["{polarization}"]
    }}],
    output: {{
      bands: 1,
      sampleType: "FLOAT32"
    }}
  }};
}}

function evaluatePixel(sample) {{
  return [sample.{polarization}];
}}
"""

            payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                    },
                    "data": [{
                        "type": "sentinel-1-grd",
                        "dataFilter": {
                            "timeRange": {
                                "from": time_range["from"],
                                "to": time_range["to"]
                            },
                            "orbitDirection": "ASCENDING"
                        },
                        "processing": {
                            "backCoeff": "SIGMA0_ELLIPSOID",
                            "orthorectify": True,
                            "speckleFilter": {
                                "type": "LEE",
                                "windowSizeX": 5,
                                "windowSizeY": 5
                            }
                        }
                    }]
                },
                "output": {
                    "width": width,
                    "height": height,
                    "responses": [{
                        "identifier": "default",
                        "format": {"type": "image/tiff"}
                    }]
                },
                "evalscript": evalscript
            }

            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "image/tiff"
            }

            response = requests.post(url, json=payload, headers=headers, timeout=120)

            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully acquired Sentinel-1 data: {output_path}")
                return DataStatus.OK, output_path
            else:
                logger.error(f"Sentinel-1 acquisition failed: {response.status_code} - {response.text}")
                return DataStatus.API_ERROR, f"HTTP {response.status_code}"

        except Exception as e:
            logger.error(f"Sentinel-1 acquisition error: {e}")
            return DataStatus.API_ERROR, str(e)

    def acquire_sentinel2(self, bbox: List[float], time_range: Dict[str, str],
                         output_path: str, width: int = 512, height: int = 512) -> Tuple[DataStatus, str]:
        """Acquire Sentinel-2 optical RGB imagery"""
        self._ensure_token()

        try:
            evalscript = """
//VERSION=3
function setup() {
  return {
    input: [{
      bands: ["B04", "B03", "B02", "SCL"]
    }],
    output: {
      bands: 3,
      sampleType: "UINT16"
    }
  };
}

function evaluatePixel(sample) {
  // Simple cloud masking using SCL
  if (sample.SCL === 3 || sample.SCL === 8 || sample.SCL === 9) {
    return [0, 0, 0]; // Clouds/shadows
  }
  return [sample.B04, sample.B03, sample.B02];
}
"""

            payload = {
                "input": {
                    "bounds": {
                        "bbox": bbox,
                        "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
                    },
                    "data": [{
                        "type": "sentinel-2-l2a",
                        "dataFilter": {
                            "timeRange": {
                                "from": time_range["from"],
                                "to": time_range["to"]
                            },
                            "maxCloudCoverage": 30
                        }
                    }]
                },
                "output": {
                    "width": width,
                    "height": height,
                    "responses": [{
                        "identifier": "default",
                        "format": {"type": "image/tiff"}
                    }]
                },
                "evalscript": evalscript
            }

            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Type": "application/json",
                "Accept": "image/tiff"
            }

            response = requests.post(f"{self.base_url}/api/v1/process", 
                                   json=payload, headers=headers, timeout=120)

            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                logger.info(f"Successfully acquired Sentinel-2 data: {output_path}")
                return DataStatus.OK, output_path
            else:
                logger.error(f"Sentinel-2 acquisition failed: {response.status_code}")
                return DataStatus.API_ERROR, f"HTTP {response.status_code}"

        except Exception as e:
            logger.error(f"Sentinel-2 acquisition error: {e}")
            return DataStatus.API_ERROR, str(e)

class RasterValidator:
    """Validates satellite imagery rasters"""

    @staticmethod
    def validate(raster_path: str, min_size_mb: float = 0.001) -> ValidationResult:
        """
        Validate a raster file
        NO_DATA conditions:
        - File too small
        - All zeros or NaNs
        - Invalid shape
        """
        try:
            file_size_mb = os.path.getsize(raster_path) / (1024 * 1024)

            if file_size_mb < min_size_mb:
                return ValidationResult(
                    is_valid=False,
                    status=DataStatus.NO_DATA,
                    reason=f"File too small: {file_size_mb:.4f} MB",
                    file_size_mb=file_size_mb
                )

            with rasterio.open(raster_path) as src:
                shape = (src.height, src.width, src.count) if src.count > 1 else (src.height, src.width)

                if src.count == 1:
                    data = src.read(1)
                else:
                    data = src.read()

                if data.size == 0:
                    return ValidationResult(
                        is_valid=False,
                        status=DataStatus.INVALID_PRODUCT,
                        reason="Empty raster",
                        file_size_mb=file_size_mb,
                        shape=shape
                    )

                valid_mask = ~np.isnan(data) & (data != 0) if np.issubdtype(data.dtype, np.floating) else (data != 0)
                nodata_ratio = 1.0 - (np.sum(valid_mask) / data.size)

                if nodata_ratio > 0.95:
                    return ValidationResult(
                        is_valid=False,
                        status=DataStatus.NO_DATA,
                        reason=f"Mostly empty: {nodata_ratio*100:.1f}% no-data",
                        file_size_mb=file_size_mb,
                        shape=shape,
                        nodata_ratio=nodata_ratio
                    )

                valid_data = data[valid_mask]
                stats = {
                    "min": float(np.min(valid_data)) if valid_data.size > 0 else 0.0,
                    "max": float(np.max(valid_data)) if valid_data.size > 0 else 0.0,
                    "mean": float(np.mean(valid_data)) if valid_data.size > 0 else 0.0,
                    "std": float(np.std(valid_data)) if valid_data.size > 0 else 0.0
                }

                return ValidationResult(
                    is_valid=True,
                    status=DataStatus.OK,
                    reason="Valid raster",
                    file_size_mb=file_size_mb,
                    shape=shape,
                    nodata_ratio=nodata_ratio,
                    stats=stats
                )

        except Exception as e:
            return ValidationResult(
                is_valid=False,
                status=DataStatus.INVALID_PRODUCT,
                reason=f"Validation error: {str(e)}",
                file_size_mb=0.0
            )

class ImageryRenderer:
    """Renders satellite imagery to PNG visualizations"""

    @staticmethod
    def render_sar(input_path: str, output_path: str, 
                   percentile_range: Tuple[float, float] = (10, 99.8)) -> bool:
        """
        Render SAR data to grayscale PNG with dB conversion
        """
        try:
            with rasterio.open(input_path) as src:
                data = src.read(1).astype(np.float32)

                valid_mask = (data > 0) & np.isfinite(data)

                if not np.any(valid_mask):
                    logger.warning("No valid SAR data to render")
                    return False

                data_db = np.full_like(data, np.nan)
                data_db[valid_mask] = 10 * np.log10(data[valid_mask])

                valid_db = data_db[valid_mask]
                vmin, vmax = np.percentile(valid_db, percentile_range)

                normalized = np.clip((data_db - vmin) / (vmax - vmin), 0, 1)
                normalized[~valid_mask] = 0
                img_data = (normalized * 255).astype(np.uint8)

                img = Image.fromarray(img_data, mode='L')
                img.save(output_path, 'PNG')

                logger.info(f"Rendered SAR to {output_path}")
                return True

        except Exception as e:
            logger.error(f"SAR rendering failed: {e}")
            return False

    @staticmethod
    def render_optical(input_path: str, output_path: str, 
                      gamma: float = 2.2) -> bool:
        """
        Render optical RGB data to PNG with normalization
        """
        try:
            with rasterio.open(input_path) as src:

                rgb = src.read([1, 2, 3])  

                normalized = np.zeros_like(rgb, dtype=np.float32)

                for i in range(3):
                    band = rgb[i].astype(np.float32)
                    valid_mask = (band > 0) & np.isfinite(band)

                    if np.any(valid_mask):
                        valid_data = band[valid_mask]
                        p2, p98 = np.percentile(valid_data, [2, 98])

                        band_norm = np.clip((band - p2) / (p98 - p2), 0, 1)
                        band_norm[~valid_mask] = 0

                        normalized[i] = np.power(band_norm, 1/gamma)

                img_data = (normalized * 255).astype(np.uint8)
                img_data = np.transpose(img_data, (1, 2, 0))  

                img = Image.fromarray(img_data, mode='RGB')
                img.save(output_path, 'PNG')

                logger.info(f"Rendered optical to {output_path}")
                return True

        except Exception as e:
            logger.error(f"Optical rendering failed: {e}")
            return False

class NewsCollector:
    """Collects and ranks news coverage (GDELT-first)"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key  

    def collect(self, keywords: List[str], time_range: Dict[str, str],
                language: Optional[List[str]] = None, max_articles: int = 50) -> List[NewsArticle]:
        logger.info(f"Collecting news for keywords: {keywords}")

        try:
            gdelt_articles = self._collect_gdelt(
                keywords=keywords,
                time_range=time_range,
                language=language,
                max_articles=max_articles
            )
            if gdelt_articles:
                return gdelt_articles
        except Exception as e:
            logger.warning(f"GDELT collection failed, falling back. Reason: {e}")

        if self.api_key:
            try:
                newsapi_articles = self._collect_newsapi(
                    keywords=keywords,
                    time_range=time_range,
                    language=language,
                    max_articles=max_articles
                )
                if newsapi_articles:
                    return newsapi_articles
            except Exception as e:
                logger.warning(f"NewsAPI collection failed. Reason: {e}")

        logger.warning("No news sources returned results.")
        return []

    def _collect_gdelt(self, keywords: List[str], time_range: Dict[str, str],
                      language: Optional[List[str]], max_articles: int) -> List[NewsArticle]:
        """
        GDELT 2.1 DOC API
        https://api.gdeltproject.org/api/v2/doc/doc
        """

        q = " OR ".join([f'"{k}"' for k in keywords if k])
        if not q:
            q = '"incident"'

        def to_gdelt_ts(iso_str: str) -> str:

            s = iso_str.replace("Z", "+00:00")
            dt = datetime.fromisoformat(s)
            return dt.strftime("%Y%m%d%H%M%S")

        start = to_gdelt_ts(time_range["from"])
        end = to_gdelt_ts(time_range["to"])

        params = {
            "query": q,
            "mode": "ArtList",
            "format": "json",
            "startdatetime": start,
            "enddatetime": end,
            "maxrecords": min(max_articles, 250),  

            "sort": "HybridRel",
        }

        if language and len(language) > 0:
            params["format"] = "json"
            params["sourcelang"] = language[0]

        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        articles = []
        for a in data.get("articles", []):

            title = a.get("title") or ""
            source = a.get("sourceCountry") or a.get("sourceCollection") or a.get("domain") or "Unknown"
            web_url = a.get("url") or ""
            date = a.get("seendate") or a.get("datetime") or ""
            snippet = a.get("snippet") or a.get("excerpt") or ""

            credibility = 0.6
            dom = (a.get("domain") or "").lower()
            if any(x in dom for x in ["reuters.com", "apnews.com", "bbc.", "nytimes.com", "wsj.com"]):
                credibility = 0.9

            text = (title + " " + snippet).lower()
            hits = sum(1 for k in keywords if k and k.lower() in text)
            relevance = min(1.0, 0.4 + 0.15 * hits)

            articles.append(NewsArticle(
                title=title[:300],
                publisher=(a.get("domain") or source)[:120],
                url=web_url,
                date=date,
                snippet=snippet[:500],
                full_text=None,
                relevance_score=relevance,
                credibility_score=credibility,
                location_mentions=None
            ))

        logger.info(f"GDELT returned {len(articles)} articles")
        return articles

    def _collect_newsapi(self, keywords: List[str], time_range: Dict[str, str],
                        language: Optional[List[str]], max_articles: int) -> List[NewsArticle]:
        """
        NewsAPI 'everything' endpoint (requires API key)
        """
        q = " OR ".join(keywords) if keywords else "incident"
        url = "https://newsapi.org/v2/everything"

        params = {
            "q": q,
            "from": time_range["from"].replace("Z", ""),
            "to": time_range["to"].replace("Z", ""),
            "pageSize": min(max_articles, 100),
            "sortBy": "relevancy",
        }
        if language and len(language) > 0:
            params["language"] = language[0]

        headers = {"X-Api-Key": self.api_key}
        r = requests.get(url, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()

        out = []
        for a in data.get("articles", []):
            dom = (a.get("url") or "")
            publisher = (a.get("source") or {}).get("name") or "Unknown"
            title = a.get("title") or ""
            snippet = a.get("description") or ""
            date = a.get("publishedAt") or ""

            credibility = 0.6
            if publisher.lower() in ["reuters", "associated press", "ap", "bbc news", "the new york times"]:
                credibility = 0.9

            out.append(NewsArticle(
                title=title[:300],
                publisher=publisher[:120],
                url=a.get("url") or "",
                date=date,
                snippet=snippet[:500],
                full_text=None,
                relevance_score=0.7,
                credibility_score=credibility,
                location_mentions=None
            ))

        logger.info(f"NewsAPI returned {len(out)} articles")
        return out

    def rank_articles(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        def score(article: NewsArticle) -> float:
            return article.relevance_score * 0.5 + article.credibility_score * 0.3 + 0.2
        return sorted(articles, key=score, reverse=True)

class EventExtractor:
    """Extracts structured event claims from news"""

    def extract_claims(self, articles: List[NewsArticle]) -> List[EventClaim]:
        """
        Extract event claims from articles
        In production, use NLP/LLM for extraction
        """
        claims = []

        for article in articles:

            claim = EventClaim(
                what="Reported incident",
                where="AOI region",
                when=article.date,
                uncertainty="medium",
                sources=[article.url]
            )
            claims.append(claim)

        return claims


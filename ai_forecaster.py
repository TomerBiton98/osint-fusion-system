import logging
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from transformers import pipeline
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face not available – running in fallback mode")

class AIForecaster:
    def __init__(self, risk_profile: str = "medium"):
        self.risk_profile = risk_profile
        self.models = {}
        self.model_versions = {}

        if HF_AVAILABLE:
            self._initialize_models()

    def _initialize_models(self):
        try:
            logger.info("Loading zero-shot classifier...")
            self.models["classifier"] = pipeline(
                task="zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1,
            )
            self.model_versions["classifier"] = "facebook/bart-large-mnli"

            logger.info("Loading sentiment model...")
            self.models["sentiment"] = pipeline(
                task="sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1,
            )
            self.model_versions["sentiment"] = "cardiffnlp/twitter-roberta-base-sentiment-latest"

            logger.info("Loading summarization model (text2text-generation)...")
            self.models["summarizer"] = pipeline(
                task="summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1,
            )
            self.model_versions["summarizer"] = "facebook/bart-large-cnn"

            logger.info("All AI models loaded successfully")

        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.models = {}

    def classify_event(self, text: str) -> Dict[str, float]:
        labels = [
            "airstrike",
            "explosion",
            "fire",
            "ground operation",
            "evacuation",
            "infrastructure damage",
            "protest",
            "aid delivery",
            "cyber incident",
            "military movement",
        ]

        if "classifier" not in self.models or not text:
            return {}

        result = self.models["classifier"](text, labels, multi_label=True)
        return dict(zip(result["labels"], result["scores"]))

    def analyze_urgency(self, texts: List[str]) -> Tuple[float, str]:
        if "sentiment" not in self.models or not texts:
            return 0.5, "Fallback urgency"

        scores = []
        for t in texts[:10]:
            r = self.models["sentiment"](t[:512])[0]
            if r["label"] == "NEGATIVE":
                scores.append(r["score"])
            elif r["label"] == "POSITIVE":
                scores.append(1 - r["score"])
            else:
                scores.append(0.5)

        avg = float(np.mean(scores))

        trend = (
            "High urgency sentiment trend"
            if avg > 0.7
            else "Medium urgency sentiment trend"
            if avg > 0.4
            else "Low urgency sentiment trend"
        )

        return avg, trend

    def generate_forecast(
        self,
        event_classifications: Dict[str, float],
        urgency_score: float,
        urgency_trend: str,
        num_sources: int,
        contradiction_score: float,
        satellite_evidence: Optional[str] = None,
    ) -> Dict:

        if not event_classifications:
            return {"status": "no_signal"}

        top_event, top_conf = max(event_classifications.items(), key=lambda x: x[1])

        predictions = [{
            "hypothesis": f"Follow-on activity related to {top_event}",
            "confidence": self._calculate_confidence(
                top_conf, urgency_score, num_sources, contradiction_score
            ),
            "signals": [
                f"Top event: {top_event} ({top_conf:.2f})",
                urgency_trend,
                f"{num_sources} sources",
                "Satellite confirmation available" if satellite_evidence else "No satellite confirmation",
            ],
            "what_to_verify_next": generate_verification_checklist(
                top_event, satellite_evidence is not None
            ),
        }]

        return {
            "status": "OK",
            "forecast_horizon_hours": 24,
            "predictions": self._adjust_for_risk_profile(predictions),
            "model_versions": self.model_versions,
            "features_used": {
                "top_event": top_event,
                "urgency_score": urgency_score,
                "num_sources": num_sources,
                "contradiction_score": contradiction_score,
            },
        }

    def _calculate_confidence(self, event, urgency, sources, contradiction):
        src_factor = min(sources / 5, 1.0)
        conf = (
            event * 0.4 +
            urgency * 0.2 +
            src_factor * 0.2 +
            (1 - contradiction) * 0.2
        )

        cap = 0.75 if self.risk_profile == "high" else 0.9
        return round(min(conf, cap), 2)

    def _adjust_for_risk_profile(self, predictions):
        if self.risk_profile == "high":
            for p in predictions:
                p["confidence"] = round(p["confidence"] * 0.85, 2)
                p["signals"].append("Confidence reduced due to HIGH risk profile")
        return predictions

    def summarize_news(self, texts: List[str], max_tokens: int = 120) -> str:
        """
        Safe summarization for transformers v5+
        """
        if "summarizer" not in self.models or not texts:
            return texts[0][:300] + "..." if texts else "No data"

        joined = " ".join(texts[:5])[:3000]

        result = self.models["summarizer"](
            joined,
            max_new_tokens=max_tokens,
            do_sample=False,
        )

        return result[0]["generated_text"]

def generate_verification_checklist(event_type: str, satellite_available: bool) -> List[str]:
    base = [
        "Cross-check multiple independent sources",
        "Verify timestamps and location consistency",
        "Assess contradiction level",
    ]

    if satellite_available:
        base.extend([
            "Compare pre/post satellite imagery",
            "Look for structural or terrain changes",
            "Check SAR backscatter anomalies",
        ])
    else:
        base.append("No satellite imagery available")

    return base


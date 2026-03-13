import logging
from datetime import datetime, timedelta

from pipeline import OSINTFusionPipeline
from osint_fusion_system import ImageryRequest, SatelliteMode, RiskProfile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main():
    print("Initializing OSINT Fusion Pipeline...")

    pipeline = OSINTFusionPipeline(output_dir="./my_analysis")

    request = ImageryRequest(
        bbox=[51.20, 35.50, 51.70, 35.90],

        time_range={
            "from": (datetime.utcnow() - timedelta(days=365)).isoformat() + "Z",
            "to": datetime.utcnow().isoformat() + "Z",
        },

        aoi_name="Iran",

        satellite_mode=SatelliteMode.AUTO,

        news_keywords=[

            "iran",
            "iranian",
            "tehran",
            "islamic republic",
            "ayatollah",
            "khamenei",
            "iran government",

            "iran military",
            "iran army",
            "iranian armed forces",
            "revolutionary guard",
            "IRGC",

            "missile test",
            "ballistic missile",
            "drone strike",
            "air defense",
            "military base",

            "nuclear program",
            "uranium enrichment",
            "nuclear facility",
            "IAEA",

            "sanctions",
            "oil export",
            "persian gulf",
            "strait of hormuz",

            "protests in iran",
            "civil unrest",
            "iran protests"

        ],

        language_filter=["en"],
        risk_profile=RiskProfile.MEDIUM,
    )

    print("\nStarting analysis...")
    print(f"Area: {request.aoi_name}")
    print(f"Time: {request.time_range['from']} → {request.time_range['to']}")
    print(f"Mode: {request.satellite_mode.value}")
    print("\nThis may take 1–3 minutes...\n")

    results = pipeline.run_analysis(request)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    print(f"\nStatus: {results['status']}")
    print(f"Job ID: {results['job_id']}")

    imagery = results.get("imagery", {}).get("acquisitions", {})
    if imagery:
        print("\nSatellite Imagery:")
        if imagery.get("s1"):
            print(f"  • Sentinel-1 SAR: {imagery['s1']['status']}")
            if imagery["s1"].get("visualization"):
                print(f"    Visualization: {imagery['s1']['visualization']}")

        if imagery.get("s2"):
            print(f"  • Sentinel-2 Optical: {imagery['s2']['status']}")
            if imagery["s2"].get("visualization"):
                print(f"    Visualization: {imagery['s2']['visualization']}")

    news = results.get("news", {})
    print("\nNews Coverage:")
    print(f"  • Articles collected: {news.get('article_count', 0)}")
    print(f"  • Event claims: {news.get('claim_count', 0)}")

    forecast = results.get("forecast", {})
    if forecast:
        print("\nAI Forecast:")
        print(f"  • Status: {forecast.get('status', 'ok')}")
        print(f"  • Confidence score: {forecast.get('confidence', 'N/A')}")

        preds = forecast.get("predictions", [])
        if preds:
            print(f"  • Predictions: {len(preds)}")
            for i, p in enumerate(preds, 1):
                print(f"\n  Prediction {i}:")
                print(f"    Hypothesis: {p['hypothesis']}")
                print(f"    Confidence: {p['confidence']}")

    if results.get("report_md"):
        print("\nReports Generated:")
        print(f"  • Markdown report: {results['report_md']}")
        print(f"  • JSON report: {results.get('results_json', 'N/A')}")

    print("\n" + "=" * 70)
    print(f"Analysis saved to: ./my_analysis/{results['job_id']}/")
    print("\nNext steps:")
    print("  1. Review the markdown report")
    print("  2. Inspect satellite imagery")
    print("  3. Evaluate confidence score")
    print("  4. Cross-reference news vs imagery")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nERROR: {e}")
        print("\nChecklist:")
        print("  ✓ .env loaded")
        print("  ✓ Copernicus credentials set")
        print("  ✓ RSS feeds reachable")
        print("  ✓ requirements.txt installed")


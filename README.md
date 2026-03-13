# OSINT Fusion Intelligence System

A production-style OSINT intelligence platform that combines satellite imagery, news intelligence, and AI forecasting to generate verified intelligence reports.

## Overview

This system fuses three sources of intelligence:

• Satellite imagery (Copernicus Sentinel-1 & Sentinel-2)  
• News coverage from multiple media sources  
• AI-based event classification and forecasting  

The platform generates structured intelligence reports with explicit uncertainty and verification recommendations.

---

## Architecture

```
User Request
      ↓
Pipeline Orchestrator
      ↓
Satellite Acquisition (Sentinel-1 / Sentinel-2)
      ↓
Raster Validation
      ↓
News Collection
      ↓
Event Extraction
      ↓
AI Forecasting
      ↓
Fusion Intelligence Report
```

---

## Features

• Sentinel-1 SAR satellite acquisition  
• Sentinel-2 optical imagery  
• automatic raster validation  
• multi-source news intelligence collection  
• event extraction and claim verification  
• AI event classification (HuggingFace models)  
• urgency analysis and forecasting  
• REST API via FastAPI  

---

## Installation

```bash
git clone https://github.com/yourname/osint-fusion-system
cd osint-fusion-system

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

---

## Configuration

Create a `.env` file:

```
SH_CLIENT_ID=your_client_id
SH_CLIENT_SECRET=your_client_secret
NYT_API_KEY=optional
```

---

## Usage

### Run analysis

```python
from pipeline import OSINTFusionPipeline
from osint_fusion_system import ImageryRequest

pipeline = OSINTFusionPipeline()

request = ImageryRequest(
    bbox=[34.40, 31.40, 34.55, 31.55],
    time_range={
        "from": "2024-01-20T00:00:00Z",
        "to": "2024-01-27T23:59:59Z"
    },
    news_keywords=["incident","explosion"]
)

results = pipeline.run_analysis(request)
```


## Tech Stack

Python  
FastAPI  
HuggingFace Transformers  
NumPy  
Rasterio  
Copernicus Dataspace API  

---

## Project Status

Prototype / research project exploring automated OSINT intelligence fusion.

---

## Author

Tomer Biton  
AI & Data Systems Builder
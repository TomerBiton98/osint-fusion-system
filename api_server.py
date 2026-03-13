from fastapi import FastAPI, HTTPException, BackgroundTasks, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from pathlib import Path
import json
import logging
from datetime import datetime
from enum import Enum

from pipeline import OSINTFusionPipeline
from osint_fusion_system import ImageryRequest, SatelliteMode, OrbitDirection, RiskProfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="OSINT Fusion Intelligence API",
    description="Production-grade OSINT platform fusing satellite imagery with news coverage and AI forecasting",
    version="1.0.0"
)

pipeline = OSINTFusionPipeline(output_dir="./api_outputs")

jobs_db: Dict[str, Dict] = {}

class TimeRange(BaseModel):
    from_time: str = Field(..., alias="from", description="Start time (ISO 8601)")
    to_time: str = Field(..., alias="to", description="End time (ISO 8601)")

class AnalysisRequest(BaseModel):
    bbox: List[float] = Field(..., description="Bounding box [min_lon, min_lat, max_lon, max_lat]")
    time_range: TimeRange
    aoi_name: Optional[str] = Field(None, description="Area of interest name")
    satellite_mode: Optional[str] = Field("AUTO", description="Satellite mode: S1, S2, or AUTO")
    orbit_direction: Optional[str] = Field("ANY", description="Orbit: ASCENDING, DESCENDING, or ANY")
    news_keywords: Optional[List[str]] = Field(None, description="Keywords for news search")
    language_filter: Optional[List[str]] = Field(None, description="Language codes (e.g., ['en', 'ar'])")
    risk_profile: Optional[str] = Field("MEDIUM", description="Risk profile: LOW, MEDIUM, or HIGH")
    output_size: Optional[List[int]] = Field([2500, 2500], description="Output image size [width, height]")

    class Config:
        populate_by_name = True

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    job_id: str
    status: str
    timestamp: str
    progress: Optional[Dict] = None
    errors: Optional[List[Dict]] = None
    results: Optional[Dict] = None

@app.get("/")
async def root():
    """API health check"""
    return {
        "service": "OSINT Fusion Intelligence API",
        "status": "operational",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "components": {
            "pipeline": "initialized",
            "jobs_count": len(jobs_db)
        },
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/run", response_model=JobResponse)
async def run_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Submit a new OSINT fusion analysis job
    Returns immediately with job_id
    Analysis runs in background
    """
    try:

        imagery_request = ImageryRequest(
            bbox=request.bbox,
            time_range={
                "from": request.time_range.from_time,
                "to": request.time_range.to_time
            },
            aoi_name=request.aoi_name,
            satellite_mode=SatelliteMode[request.satellite_mode.upper()],
            orbit_direction=OrbitDirection[request.orbit_direction.upper()],
            news_keywords=request.news_keywords,
            language_filter=request.language_filter,
            risk_profile=RiskProfile[request.risk_profile.upper()],
            output_size=tuple(request.output_size)
        )

        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        jobs_db[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "timestamp": datetime.now().isoformat(),
            "request": request.dict()
        }

        background_tasks.add_task(run_analysis_job, job_id, imagery_request)

        logger.info(f"Queued analysis job: {job_id}")

        return JobResponse(
            job_id=job_id,
            status="queued",
            message=f"Analysis job {job_id} has been queued and will begin processing shortly"
        )

    except Exception as e:
        logger.error(f"Error creating job: {e}")
        raise HTTPException(status_code=400, detail=str(e))

async def run_analysis_job(job_id: str, request: ImageryRequest):
    """Background task to run analysis"""
    try:
        jobs_db[job_id]["status"] = "running"
        logger.info(f"Starting analysis job: {job_id}")

        results = pipeline.run_analysis(request)

        jobs_db[job_id]["status"] = results["status"]
        jobs_db[job_id]["results"] = results
        jobs_db[job_id]["timestamp_completed"] = datetime.now().isoformat()

        logger.info(f"Completed analysis job: {job_id} with status {results['status']}")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)

@app.get("/api/job/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """
    Get status and results of an analysis job
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = jobs_db[job_id]

    return JobStatus(
        job_id=job_data["job_id"],
        status=job_data["status"],
        timestamp=job_data["timestamp"],
        errors=job_data.get("results", {}).get("errors"),
        results=job_data.get("results") if job_data["status"] == "completed" else None
    )

@app.get("/api/jobs")
async def list_jobs(limit: int = 10):
    """List recent jobs"""
    jobs_list = list(jobs_db.values())
    jobs_list.sort(key=lambda x: x["timestamp"], reverse=True)

    return {
        "jobs": jobs_list[:limit],
        "total": len(jobs_db)
    }

@app.get("/api/output/{job_id}/file")
async def get_output_file(job_id: str, file_type: str):
    """
    Download output files from a job
    file_type: 'report_md', 'report_json', 's1_viz', 's2_viz', 'news_raw', 'news_ranked', 'results'
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = jobs_db[job_id]

    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} is not completed (status: {job_data['status']})")

    results = job_data.get("results", {})

    file_paths = {
        "report_md": results.get("report_md"),
        "report_json": results.get("report_json"),
        "s1_viz": results.get("imagery", {}).get("acquisitions", {}).get("s1", {}).get("visualization"),
        "s2_viz": results.get("imagery", {}).get("acquisitions", {}).get("s2", {}).get("visualization"),
        "news_raw": results.get("news", {}).get("raw_path"),
        "news_ranked": results.get("news", {}).get("ranked_path"),
        "results": str(Path(pipeline.output_dir) / results["job_id"] / "results.json")
    }

    file_path = file_paths.get(file_type)

    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail=f"File type '{file_type}' not found for job {job_id}")

    media_types = {
        ".md": "text/markdown",
        ".json": "application/json",
        ".png": "image/png",
        ".tif": "image/tiff"
    }

    file_ext = Path(file_path).suffix
    media_type = media_types.get(file_ext, "application/octet-stream")

    return FileResponse(
        file_path,
        media_type=media_type,
        filename=Path(file_path).name
    )

@app.get("/api/output/{job_id}/report")
async def get_report(job_id: str):
    """Get the markdown report as JSON"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = jobs_db[job_id]

    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} not completed")

    report_path = job_data.get("results", {}).get("report_md")

    if not report_path or not Path(report_path).exists():
        raise HTTPException(status_code=404, detail="Report not found")

    with open(report_path, 'r') as f:
        report_content = f.read()

    return {
        "job_id": job_id,
        "report_type": "markdown",
        "content": report_content
    }

@app.get("/api/output/{job_id}/forecast")
async def get_forecast(job_id: str):
    """Get the AI forecast"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    job_data = jobs_db[job_id]

    if job_data["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job {job_id} not completed")

    forecast = job_data.get("results", {}).get("forecast")

    if not forecast:
        raise HTTPException(status_code=404, detail="Forecast not found")

    return forecast

@app.delete("/api/job/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its outputs"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    del jobs_db[job_id]

    return {"message": f"Job {job_id} deleted"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


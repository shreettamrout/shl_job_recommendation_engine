import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------------------------------
# Ensure src/ is importable
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.pipeline import AssessmentRecommenderPipeline

# -------------------------------------------------
# FastAPI app
# -------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommendation API",
    description="Retrieval-Augmented GenAI system for SHL assessment recommendation",
    version="1.0"
)

# -------------------------------------------------
# Load pipeline once at startup
# -------------------------------------------------
pipeline = AssessmentRecommenderPipeline()


# -------------------------------------------------
# Request schema
# -------------------------------------------------
class RecommendationRequest(BaseModel):
    query: str


# -------------------------------------------------
# Health check
# -------------------------------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


# -------------------------------------------------
# Recommendation endpoint
# -------------------------------------------------
@app.post("/recommend")
def recommend(req: RecommendationRequest):
    recommendations = pipeline.recommend(req.query)
    return {
        "query": req.query,
        "recommendations": recommendations
    }


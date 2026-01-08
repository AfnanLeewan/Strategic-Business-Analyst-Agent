"""
FastAPI Backend for StratAI.
Provides REST API endpoints for document upload and strategic analysis.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd

from backend.core.config import settings
from backend.services.ingest import document_processor
from backend.services.rag_chain import rag_engine
from backend.services.predictor import sales_predictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="StratAI API",
    description="Strategic Business Analyst Agent - RAG-based AI backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure upload directory exists
os.makedirs(settings.upload_dir, exist_ok=True)


# Request/Response Models
class AnalyzeRequest(BaseModel):
    """Request model for analysis endpoint."""
    query: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are our company's main strengths based on the annual report?"
            }
        }


class AnalyzeResponse(BaseModel):
    """Response model for analysis endpoint."""
    success: bool
    query: str
    intent: str
    analysis: str
    sources: list
    metadata: Dict[str, Any]
    forecast: Dict[str, Any] = None


class UploadResponse(BaseModel):
    """Response model for upload endpoint."""
    success: bool
    file_name: str
    file_type: str
    chunks_created: int
    message: str
    forecast: Dict[str, Any] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    vector_db_status: Dict[str, Any]


# API Endpoints

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "StratAI API - Strategic Business Analyst Agent",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "upload": "/upload",
            "analyze": "/analyze",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns system status and vector DB statistics.
    """
    try:
        db_stats = document_processor.get_collection_stats()
        
        return {
            "status": "healthy",
            "vector_db_status": db_stats
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Service unhealthy: {str(e)}"
        )


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and index a document (PDF or CSV).
    
    - **file**: The document file to upload (PDF or CSV)
    
    Returns ingestion results including number of chunks created.
    """
    try:
        # Validate file type
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.csv']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported file type: {file_ext}. Only PDF and CSV are supported."
            )
        
        # Save uploaded file
        file_path = os.path.join(settings.upload_dir, file.filename)
        
        with open(file_path, "wb") as f:
            content = await file.read()
            
            # Check file size
            if len(content) > settings.max_file_size:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Max size: {settings.max_file_size / (1024*1024)}MB"
                )
            
            f.write(content)
        
        logger.info(f"File saved: {file_path}")
        
        # Process and ingest document
        result = document_processor.ingest_document(file_path)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Failed to process document")
            )
        
        # If CSV, try to generate forecast
        forecast_result = None
        if file_ext == '.csv' and result.get("has_dataframe"):
            try:
                # Try different encodings to match ingest.py behavior
                df = None
                encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
                
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except (UnicodeDecodeError, Exception):
                        continue
                
                if df is not None:
                    forecast_result = sales_predictor.forecast(df)
                    
                    if forecast_result and forecast_result.get("success"):
                        logger.info("Sales forecast generated successfully")
            except Exception as e:
                logger.warning(f"Could not generate forecast: {e}")
        
        
        # Build response - only include forecast if it exists and is successful
        response_data = {
            "success": True,
            "file_name": result["file_name"],
            "file_type": result["file_type"],
            "chunks_created": result["chunks_created"],
            "message": result["message"]
        }
        
        if forecast_result and forecast_result.get("success"):
            response_data["forecast"] = forecast_result
        
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload failed: {str(e)}"
        )


@app.post("/analyze", response_model=AnalyzeResponse, tags=["Analysis"])
async def analyze_query(request: AnalyzeRequest):
    """
    Analyze a business query using RAG pipeline.
    
    - **query**: The business question or query to analyze
    
    Returns strategic analysis with source citations.
    """
    try:
        if not request.query or len(request.query.strip()) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )
        
        logger.info(f"Analyzing query: {request.query}")
        
        # Run RAG analysis
        result = rag_engine.analyze(request.query)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result.get("message", "Analysis failed")
            )
        
        return {
            "success": True,
            "query": result["query"],
            "intent": result["intent"],
            "analysis": result["analysis"],
            "sources": result["sources"],
            "metadata": result["metadata"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# Run with: uvicorn backend.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.backend_host,
        port=settings.backend_port,
        reload=True
    )

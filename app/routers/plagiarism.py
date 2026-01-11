from fastapi import APIRouter, HTTPException, status
from typing import Dict
import time
import logging

from app.models import (
    PlagiarismCheckRequest,
    PlagiarismCheckResponse,
    HealthCheckResponse
)
from app.services import PlagiarismService
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/plagiarism",
    tags=["Plagiarism Detection"]
)

# Initialize plagiarism service
plagiarism_service = PlagiarismService()


@router.post(
    "/check",
    response_model=PlagiarismCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Check plagiarism across multiple assignments",
    description="""
    Analyzes multiple student assignments for plagiarism by:
    1. Generating embeddings for each assignment text
    2. Comparing all assignments with each other
    3. Calculating similarity scores
    4. Assigning marks and status based on similarity thresholds
    5. Marking earliest submission as original when high similarity detected
    
    Returns detailed plagiarism results for each assignment.
    """
)
async def check_plagiarism(request: PlagiarismCheckRequest):
    """
    Check plagiarism for a batch of assignments
    """
    try:
        start_time = time.time()
        
        logger.info(
            f"Received plagiarism check request for assignment group: "
            f"{request.assignment_group_id} with {len(request.assignments)} assignments"
        )
        
        # Validate minimum assignments
        if len(request.assignments) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="At least 2 assignments are required for plagiarism detection"
            )
        
        # Validate text content
        for assignment in request.assignments:
            if not assignment.extracted_text or len(assignment.extracted_text.strip()) < settings.MIN_TEXT_LENGTH:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Assignment {assignment.assignment_id} has insufficient text content "
                           f"(minimum {settings.MIN_TEXT_LENGTH} characters required)"
                )
        
        # Check for duplicate assignment IDs
        assignment_ids = [a.assignment_id for a in request.assignments]
        if len(assignment_ids) != len(set(assignment_ids)):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Duplicate assignment IDs detected in the request"
            )
        
        # Perform plagiarism check
        logger.info("Starting plagiarism detection...")
        results = plagiarism_service.check_plagiarism(
            assignments=request.assignments,
            assignment_group_id=request.assignment_group_id
        )
        
        processing_time = round(time.time() - start_time, 2)
        
        logger.info(
            f"Plagiarism check completed in {processing_time}s. "
            f"Processed {len(results)} assignments"
        )
        
        # Log summary statistics
        copied_count = sum(1 for r in results if r.status == "COPIED")
        high_risk_count = sum(1 for r in results if r.status in ["HIGH RISK", "VERY HIGH RISK"])
        
        if copied_count > 0 or high_risk_count > 0:
            logger.warning(
                f"Plagiarism detected: {copied_count} copied, {high_risk_count} high risk"
            )
        
        return PlagiarismCheckResponse(
            success=True,
            assignment_group_id=request.assignment_group_id,
            total_assignments_checked=len(results),
            results=results,
            processing_time_seconds=processing_time
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error in plagiarism check endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during plagiarism check: {str(e)}"
        )


@router.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check endpoint",
    description="Check if the plagiarism detection service is healthy and operational"
)
async def health_check():
    """
    Health check endpoint to verify service status
    """
    try:
        health_status = plagiarism_service.health_check()
        
        if health_status["status"] == "healthy":
            return HealthCheckResponse(
                status="healthy",
                service=settings.APP_NAME,
                version=settings.APP_VERSION,
                chroma_db_status="connected"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service unhealthy: {health_status.get('reason', 'Unknown')}"
            )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Health check failed: {str(e)}"
        )


@router.get(
    "/info",
    response_model=Dict,
    status_code=status.HTTP_200_OK,
    summary="Get service information",
    description="Get information about the plagiarism detection service configuration"
)
async def get_service_info():
    """
    Get service configuration and threshold information
    """
    return {
        "service_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "embedding_model": settings.EMBEDDING_MODEL,
        "thresholds": {
            "original": f"< {settings.THRESHOLD_ORIGINAL}",
            "very_low_risk": f"{settings.THRESHOLD_ORIGINAL} - {settings.THRESHOLD_VERY_LOW}",
            "low_risk": f"{settings.THRESHOLD_VERY_LOW} - {settings.THRESHOLD_LOW}",
            "suspicious": f"{settings.THRESHOLD_LOW} - {settings.THRESHOLD_SUSPICIOUS}",
            "high_risk": f"{settings.THRESHOLD_SUSPICIOUS} - {settings.THRESHOLD_HIGH}",
            "very_high_risk": f"{settings.THRESHOLD_HIGH} - {settings.THRESHOLD_VERY_HIGH}",
            "copied": f">= {settings.THRESHOLD_VERY_HIGH}"
        },
        "scoring": {
            "original": "10 marks",
            "very_low_risk": "9 marks",
            "low_risk": "8 marks",
            "suspicious": "6-7 marks",
            "high_risk": "4-5 marks",
            "very_high_risk": "2-3 marks",
            "copied": "0-1 marks"
        },
        "min_text_length": settings.MIN_TEXT_LENGTH
    }
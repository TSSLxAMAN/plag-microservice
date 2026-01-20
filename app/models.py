from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class PlagiarismStatus(str, Enum):
    ORIGINAL = "ORIGINAL"
    VERY_LOW_RISK = "VERY LOW RISK"
    LOW_RISK = "LOW RISK"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH RISK"
    VERY_HIGH_RISK = "VERY HIGH RISK"
    COPIED = "COPIED"


class AssignmentInput(BaseModel):
    assignment_id: str = Field(..., description="Unique assignment submission ID")
    extracted_text: str = Field(..., description="Extracted text from student assignment")
    submitted_at: datetime = Field(..., description="Submission timestamp")
    student_id: str = Field(..., description="Student ID for reference")
    
    class Config:
        json_schema_extra = {
            "example": {
                "assignment_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "extracted_text": "This is the student's assignment text...",
                "submitted_at": "2024-01-10T10:30:00",
                "student_id": "student123"
            }
        }


class PlagiarismCheckRequest(BaseModel):
    assignments: List[AssignmentInput] = Field(
        ..., 
        min_length=2,
        description="List of assignments to check for plagiarism"
    )
    assignment_group_id: str = Field(
        ..., 
        description="Assignment ID to group submissions (e.g., all submissions for Assignment 1)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "assignment_group_id": "assignment_001",
                "assignments": [
                    {
                        "assignment_id": "sub_001",
                        "extracted_text": "This is student A's assignment...",
                        "submitted_at": "2024-01-10T10:30:00",
                        "student_id": "student_a"
                    },
                    {
                        "assignment_id": "sub_002",
                        "extracted_text": "This is student B's assignment...",
                        "submitted_at": "2024-01-10T11:00:00",
                        "student_id": "student_b"
                    }
                ]
            }
        }


class SimilarityDetail(BaseModel):
    compared_with_assignment_id: str
    compared_with_student_id: str
    similarity_score: float
    submitted_earlier: bool = Field(
        description="Whether the compared assignment was submitted earlier"
    )


class AssignmentResult(BaseModel):
    assignment_id: str
    student_id: str

    max_similarity: float = Field(
        description="Maximum similarity score found (0-1)"
    )

    plagiarism_score: float = Field(
        ge=0, le=1,
        description="Plagiarism penalty score (0 = clean, 1 = fully plagiarized)"
    )

    marks: float = Field(ge=0, le=10)
    status: PlagiarismStatus
    most_similar_to: Optional[str] = None
    similarity_details: List[SimilarityDetail] = []
    message: Optional[str] = None



class PlagiarismCheckResponse(BaseModel):
    success: bool
    assignment_group_id: str
    total_assignments_checked: int
    results: List[AssignmentResult]
    processing_time_seconds: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "assignment_group_id": "assignment_001",
                "total_assignments_checked": 2,
                "processing_time_seconds": 1.23,
                "results": [
                    {
                        "assignment_id": "sub_001",
                        "student_id": "student_a",
                        "plagiarism_score": 0.65,
                        "marks": 10.0,
                        "status": "ORIGINAL",
                        "most_similar_to": None,
                        "similarity_details": [],
                        "message": "Original work"
                    }
                ]
            }
        }


class HealthCheckResponse(BaseModel):
    status: str
    service: str
    version: str
    chroma_db_status: str
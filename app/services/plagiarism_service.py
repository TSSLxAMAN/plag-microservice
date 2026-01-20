import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime
import logging

from app.config import settings
from app.models import (
    AssignmentInput,
    AssignmentResult,
    PlagiarismStatus,
    SimilarityDetail
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlagiarismService:
    def __init__(self):
        """Initialize ChromaDB and embedding model"""
        self.model = None
        self.chroma_client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize the embedding model and ChromaDB"""
        try:
            # Initialize sentence transformer model
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded successfully")
            
            # Initialize ChromaDB client
            logger.info(f"Initializing ChromaDB at: {settings.CHROMA_DB_PATH}")
            self.chroma_client = chromadb.PersistentClient(
                path=settings.CHROMA_DB_PATH,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info("ChromaDB initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PlagiarismService: {str(e)}")
            raise
    
    def get_or_create_collection(self, collection_name: str):
        """Get or create a ChromaDB collection for a specific assignment group"""
        try:
            collection = self.chroma_client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )
            logger.info(f"Collection '{collection_name}' ready")
            return collection
        except Exception as e:
            logger.error(f"Error creating collection: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for given text"""
        try:
            # Clean and validate text
            text = text.strip()
            if len(text) < settings.MIN_TEXT_LENGTH:
                logger.warning(f"Text too short: {len(text)} characters")
                return None
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return None
    
    def calculate_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            return float(similarity)
        
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def get_plagiarism_status_and_marks(
        self, 
        similarity_score: float, 
        is_earliest: bool
    ) -> Tuple[PlagiarismStatus, float, str]:
        """
        Determine plagiarism status and marks based on similarity score
        
        Returns: (status, marks, message)
        """
        # If this is the earliest submission with high similarity, mark as original
        if is_earliest and similarity_score >= settings.THRESHOLD_VERY_HIGH:
            return PlagiarismStatus.ORIGINAL, 10.0, "Original submission (submitted first)"
        
        # Apply thresholds
        if similarity_score < settings.THRESHOLD_ORIGINAL:
            return PlagiarismStatus.ORIGINAL, 10.0, "Original work"
        
        elif similarity_score < settings.THRESHOLD_VERY_LOW:
            return PlagiarismStatus.VERY_LOW_RISK, 9.0, "Very low plagiarism risk"
        
        elif similarity_score < settings.THRESHOLD_LOW:
            return PlagiarismStatus.LOW_RISK, 8.0, "Low plagiarism risk"
        
        elif similarity_score < settings.THRESHOLD_SUSPICIOUS:
            # Assign marks between 6-7 based on position in range
            marks = 7.0 - (similarity_score - settings.THRESHOLD_LOW) / \
                    (settings.THRESHOLD_SUSPICIOUS - settings.THRESHOLD_LOW)
            marks = round(marks, 1)
            return PlagiarismStatus.SUSPICIOUS, marks, "Suspicious similarity detected"
        
        elif similarity_score < settings.THRESHOLD_HIGH:
            # Assign marks between 4-5
            marks = 5.0 - (similarity_score - settings.THRESHOLD_SUSPICIOUS) / \
                    (settings.THRESHOLD_HIGH - settings.THRESHOLD_SUSPICIOUS)
            marks = round(marks, 1)
            return PlagiarismStatus.HIGH_RISK, marks, "High plagiarism risk"
        
        elif similarity_score < settings.THRESHOLD_VERY_HIGH:
            # Assign marks between 2-3
            marks = 3.0 - (similarity_score - settings.THRESHOLD_HIGH) / \
                    (settings.THRESHOLD_VERY_HIGH - settings.THRESHOLD_HIGH)
            marks = round(marks, 1)
            return PlagiarismStatus.VERY_HIGH_RISK, marks, "Very high plagiarism risk"
        
        else:
            return PlagiarismStatus.COPIED, 0.0, "Plagiarism detected"

    
    def check_plagiarism(
        self, 
        assignments: List[AssignmentInput],
        assignment_group_id: str
    ) -> List[AssignmentResult]:
        """
        Main method to check plagiarism across multiple assignments
        """
        try:
            # Sort assignments by submission time
            sorted_assignments = sorted(assignments, key=lambda x: x.submitted_at)
            
            # Generate embeddings for all assignments
            embeddings_data = []
            for assignment in sorted_assignments:
                embedding = self.generate_embedding(assignment.extracted_text)
                if embedding:
                    embeddings_data.append({
                        "assignment": assignment,
                        "embedding": embedding
                    })
                else:
                    logger.warning(
                        f"Failed to generate embedding for assignment: {assignment.assignment_id}"
                    )
            
            if len(embeddings_data) < 2:
                logger.warning("Not enough valid assignments to compare")
                # Return all as original if less than 2
                return [
                    AssignmentResult(
                        assignment_id=a.assignment_id,
                        student_id=a.student_id,
                        plagiarism_score=0.0,
                        marks=10.0,
                        status=PlagiarismStatus.ORIGINAL,
                        message="Insufficient assignments for comparison"
                    )
                    for a in assignments
                ]
            
            # Compare each assignment with all others
            results = []
            
            for i, data_i in enumerate(embeddings_data):
                assignment_i = data_i["assignment"]
                embedding_i = data_i["embedding"]
                
                max_similarity = 0.0
                most_similar_id = None
                similarity_details = []
                
                # Compare with all other assignments
                for j, data_j in enumerate(embeddings_data):
                    if i == j:
                        continue
                    
                    assignment_j = data_j["assignment"]
                    embedding_j = data_j["embedding"]
                    
                    # Calculate similarity
                    similarity = self.calculate_similarity(embedding_i, embedding_j)
                    
                    # Store similarity detail
                    similarity_detail = SimilarityDetail(
                        compared_with_assignment_id=assignment_j.assignment_id,
                        compared_with_student_id=assignment_j.student_id,
                        similarity_score=round(similarity, 4),
                        submitted_earlier=(assignment_j.submitted_at < assignment_i.submitted_at)
                    )
                    similarity_details.append(similarity_detail)
                    
                    # Track maximum similarity
                    if similarity > max_similarity:
                        max_similarity = similarity
                        most_similar_id = assignment_j.assignment_id
                
                # Determine if this is the earliest among highly similar submissions
                is_earliest = True
                if max_similarity >= settings.THRESHOLD_SUSPICIOUS:
                    # Check if any similar assignment was submitted earlier
                    for detail in similarity_details:
                        if (detail.similarity_score >= settings.THRESHOLD_VERY_HIGH and 
                            detail.submitted_earlier):
                            is_earliest = False
                            break
                
                # Get status and marks
                status, marks, message = self.get_plagiarism_status_and_marks(
                    max_similarity, 
                    is_earliest
                )
                
                # Sort similarity details by score (descending)
                similarity_details.sort(key=lambda x: x.similarity_score, reverse=True)
                penalty = self.similarity_to_penalty(max_similarity)

                result = AssignmentResult(
                    assignment_id=assignment_i.assignment_id,
                    student_id=assignment_i.student_id,
                    max_similarity=round(max_similarity, 4),
                    plagiarism_score=penalty,
                    marks=marks,
                    status=status,
                    most_similar_to = (
                        most_similar_id 
                        if status not in [PlagiarismStatus.ORIGINAL, PlagiarismStatus.VERY_LOW_RISK]
                        else None
                    ),
                    similarity_details=similarity_details[:5],
                    message=message
                )

                results.append(result)
            
            return results
        
        except Exception as e:
            logger.error(f"Error in plagiarism check: {str(e)}")
            raise
    
    def health_check(self) -> Dict[str, str]:
        """Check if service is healthy"""
        try:
            # Check if model is loaded
            if self.model is None:
                return {"status": "unhealthy", "reason": "Model not loaded"}
            
            # Check if ChromaDB is accessible
            if self.chroma_client is None:
                return {"status": "unhealthy", "reason": "ChromaDB not initialized"}
            
            # Try to list collections
            collections = self.chroma_client.list_collections()
            
            return {
                "status": "healthy",
                "model": settings.EMBEDDING_MODEL,
                "collections_count": str(len(collections))
            }
        
        except Exception as e:
            return {"status": "unhealthy", "reason": str(e)}
    
    def similarity_to_penalty(self, similarity: float) -> float:
        """
        Converts similarity score → plagiarism penalty (0-1)
        """
        if similarity < settings.THRESHOLD_LOW:
            return 0.0
        elif similarity < settings.THRESHOLD_SUSPICIOUS:
            return 0.3
        elif similarity < settings.THRESHOLD_HIGH:
            return 0.6
        elif similarity < settings.THRESHOLD_VERY_HIGH:
            return 0.8
        else:
            return 1.0

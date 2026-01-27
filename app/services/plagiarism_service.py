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
    SimilarityDetail,
    SimplifiedAssignmentResult
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
            # 🔥 FIX: Lower the threshold for the earliest submitter. 
            # If they are first and have ANY suspicious similarity, they are the 'Source' (Victim).
            # We check against THRESHOLD_SUSPICIOUS instead of THRESHOLD_VERY_HIGH.
            if is_earliest and similarity_score >= settings.THRESHOLD_SUSPICIOUS:
                return PlagiarismStatus.ORIGINAL, 10.0, "Original submission (Source of potential copies)"
            
            # Apply standard thresholds
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
) -> List[SimplifiedAssignmentResult]:
        """
        Check plagiarism and return only scores and status.
        """
        try:
            # 1. Sort by submission time (Critical for determining original author)
            sorted_assignments = sorted(assignments, key=lambda x: x.submitted_at)
            
            # 2. Generate embeddings
            embeddings_data = []
            for assignment in sorted_assignments:
                embedding = self.generate_embedding(assignment.extracted_text)
                if embedding:
                    embeddings_data.append({
                        "assignment": assignment,
                        "embedding": embedding
                    })
            
            # Handle insufficient data
            if len(embeddings_data) < 2:
                return [
                    SimplifiedAssignmentResult(
                        assignment_id=a.assignment_id,
                        student_id=a.student_id,
                        max_similarity=0.0,
                        plagiarism_score=0.0,
                        status=PlagiarismStatus.ORIGINAL
                    )
                    for a in assignments
                ]
            
            # 3. Compare All-vs-All
            results = []
            
            for i, data_i in enumerate(embeddings_data):
                assignment_i = data_i["assignment"]
                embedding_i = data_i["embedding"]
                
                max_similarity = 0.0
                similarity_details_buffer = [] # Temp list to check "earliest" logic
                
                # Compare with all others
                for j, data_j in enumerate(embeddings_data):
                    if i == j: continue
                    
                    assignment_j = data_j["assignment"]
                    embedding_j = data_j["embedding"]
                    
                    sim = self.calculate_similarity(embedding_i, embedding_j)
                    
                    if sim > max_similarity:
                        max_similarity = sim
                    
                    # We still need this small check to determine if 'i' is the original author
                    similarity_details_buffer.append({
                        "score": sim,
                        "is_older": assignment_j.submitted_at < assignment_i.submitted_at
                    })
                
                # 4. Determine "Originality"
                # If this assignment is highly similar (>THRESHOLD) BUT exists an older submission 
                # with high similarity, then THIS one is the copy. 
                # If this is the oldest, it stays ORIGINAL.
                is_original_author = True
                if max_similarity >= settings.THRESHOLD_SUSPICIOUS:
                    for detail in similarity_details_buffer:
                        if detail["score"] >= settings.THRESHOLD_VERY_HIGH and detail["is_older"]:
                            is_original_author = False
                            break
                
                # 5. Get Status and Penalty
                # We treat 'is_earliest' logic inside the status getter
                status, _, _ = self.get_plagiarism_status_and_marks(max_similarity, is_original_author)
                penalty = self.similarity_to_penalty(max_similarity)

                # Override penalty for the original author
                if status == PlagiarismStatus.ORIGINAL:
                    penalty = 0.0
                    max_similarity = 0.0 # Optional: Set sim to 0 for the original source to avoid confusion

                # 6. Construct Simplified Result
                result = SimplifiedAssignmentResult(
                    assignment_id=assignment_i.assignment_id,
                    student_id=assignment_i.student_id,
                    max_similarity=round(max_similarity, 4),
                    plagiarism_score=penalty,
                    status=status
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

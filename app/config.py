from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "Plagiarism Detection Microservice"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8002
    
    # ChromaDB Settings
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "assignments_embeddings"
    
    # Embedding Model Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"  # Fast and efficient model
    # Alternative models you can use:
    # "all-mpnet-base-v2" - Better accuracy but slower
    # "paraphrase-MiniLM-L6-v2" - Good for paraphrase detection
    
    # Plagiarism Thresholds
    THRESHOLD_ORIGINAL: float = 0.70
    THRESHOLD_VERY_LOW: float = 0.80
    THRESHOLD_LOW: float = 0.85
    THRESHOLD_SUSPICIOUS: float = 0.90
    THRESHOLD_HIGH: float = 0.93
    THRESHOLD_VERY_HIGH: float = 0.96
    
    # Scoring
    MIN_TEXT_LENGTH: int = 50  # Minimum characters for valid comparison
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
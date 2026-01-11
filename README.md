# Plagiarism Detection Microservice

A FastAPI-based microservice for detecting plagiarism in student assignments using ChromaDB and sentence embeddings.

## Features
- Text embedding generation using sentence-transformers
- Similarity comparison using ChromaDB
- Plagiarism scoring and classification
- Temporal ordering (first submission marked as original)

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the service: `uvicorn app.main:app --reload --port 8002`

## API Endpoints
- POST `/api/plagiarism/check` - Check plagiarism for multiple assignments
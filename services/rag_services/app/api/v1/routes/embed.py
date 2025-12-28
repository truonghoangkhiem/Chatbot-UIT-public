# app/api/v1/routes/embed.py
#
# Description:
# This module implements the endpoint for generating text embeddings.
# It uses a sentence-transformer model to convert text into vector representations.

from fastapi import APIRouter, HTTPException
from app.api.schemas.embed import EmbedRequest, EmbedResponse
from app.config.settings import settings
from sentence_transformers import SentenceTransformer

# Create an API router for the embed functionality
router = APIRouter(tags=["embed"])
_model = None # A global variable to cache the loaded model

def get_model():
    """
    Loads and caches the sentence-transformer model using a singleton pattern.
    This avoids reloading the model on every API request.
    
    Returns:
        SentenceTransformer: The loaded embedding model.
    """
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.emb_model)
    return _model

@router.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    """
    Endpoint to convert a list of texts into vector embeddings.

    Args:
        req (EmbedRequest): The request containing a list of texts.

    Returns:
        EmbedResponse: The list of generated vectors.
    """
    if len(req.texts) > 64:
        raise HTTPException(status_code=400, detail="Max 64 texts per request for demo.")
    
    model = get_model()
    vectors = model.encode(req.texts, normalize_embeddings=True).tolist()
    
    return EmbedResponse(vectors=vectors)
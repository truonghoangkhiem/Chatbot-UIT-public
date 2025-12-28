# app/api/v1/routes/health.py
#
# Description:
# This module provides a simple health check endpoint.
# It is used to verify that the service is running and responsive.

from fastapi import APIRouter

# Create an API router for the health check
router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    """
    Health check endpoint. Returns a simple status message.

    Returns:
        dict: A dictionary with an "ok" status.
    """
    return {"status": "ok"}
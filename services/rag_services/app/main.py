# main.py
#
# Description:
# This script serves as the main entry point for the RAG (Retrieval-Augmented Generation) service.
# It initializes the FastAPI application, configures necessary middleware such as CORS,
# and includes all the API routers from their respective modules.
#
# Key Responsibilities:
# - Instantiate the FastAPI application.
# - Configure Cross-Origin Resource Sharing (CORS) to allow requests from any origin.
# - Include API routers for health checks, embedding, search, and admin functionalities.
# - Define a root endpoint for basic service information.
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import application-specific settings and routers
from app.config.settings import settings
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.embed import router as embed_router
from app.api.v1.routes.search import router as search_router
from app.api.v1.routes.admin import router as admin_router
from app.api.v1.routes.opensearch import router as opensearch_router
from app.api.v1.routes.extraction import router as extraction_router  # KG Extraction Pipeline
from app.api.endpoints.health import router as health_v2_router  # Week 2 comprehensive health checks

app = FastAPI(title="RAG Service", version="0.1.0")

# --- Middleware Configuration ---
# Configure CORS middleware to define which origins, methods, and headers are allowed.
# This is crucial for enabling cross-domain requests from web frontends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Router Inclusion ---
# Include the routers for different API functionalities.
# Each router is prefixed with "/v1" to version the API.
app.include_router(health_router, prefix="/v1")
app.include_router(embed_router, prefix="/v1")
app.include_router(search_router, prefix="/v1")
app.include_router(admin_router, prefix="/v1")
app.include_router(opensearch_router, prefix="/v1")
app.include_router(extraction_router, prefix="/v1")  # KG Extraction Pipeline

# Week 2: Comprehensive health checks for all dependencies
app.include_router(health_v2_router, prefix="/v2")


# --- Root Endpoint ---
@app.get("/")
def root():
    """
    Root endpoint to provide basic information about the running service.
    
    Returns:
        dict: A dictionary containing the service name and its current environment.
    """
    return {"service": "rag", "env": settings.app_env}
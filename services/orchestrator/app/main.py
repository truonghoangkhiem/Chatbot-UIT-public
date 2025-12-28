"""
Main FastAPI application for orchestrator service.

This module sets up the FastAPI application with all routes and middleware.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path

# Load .env file before anything else
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)  # Override existing env vars

from .api.routes import router as api_router
from .core.container import cleanup_container, get_container

# Configure logging
log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Log the logging level
logger.info(f"Logging level set to: {log_level}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    
    Handles startup and shutdown events for the FastAPI application.
    """
    # Startup
    logger.info("Starting orchestrator service...")
    
    # Verify required environment variables
    required_env_vars = ["OPENROUTER_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        raise RuntimeError(f"Missing required environment variables: {missing_vars}")
    
    # Initialize multi-agent orchestrator early to verify Graph Reasoning
    try:
        container = get_container()
        orchestrator = container.get_multi_agent_orchestrator()
        if orchestrator.graph_reasoning_agent:
            logger.info("✓ Graph Reasoning Agent is available")
        else:
            logger.warning("⚠ Graph Reasoning Agent is NOT available")
    except Exception as e:
        logger.warning(f"⚠ Could not initialize multi-agent orchestrator: {e}")
    
    logger.info("Orchestrator service started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down orchestrator service...")
    await cleanup_container()
    logger.info("Orchestrator service shutdown complete")


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance
    """
    app = FastAPI(
        title="Chatbot-UIT Orchestrator Service",
        description="Orchestration service that coordinates RAG retrieval and agent generation",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(api_router, prefix="/api/v1")
    
    # Root endpoint
    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with service information."""
        return {
            "service": "Chatbot-UIT Orchestrator",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "health": "/api/v1/health",
                "chat": "/api/v1/chat",
                "chat_stream": "/api/v1/chat/stream",
                "conversations": "/api/v1/conversations",
                "docs": "/docs"
            }
        }
    
    return app


# Create the application instance
app = create_app()
# app/config/settings.py
#
# Description:
# This module defines and manages all configuration settings for the RAG service.
# It uses Pydantic's BaseSettings to load configuration from environment variables
# or a .env file, providing a centralized and type-safe way to handle settings.
#
# Key Responsibilities:
# - Define the application's configuration schema.
# - Load settings from environment variables or a specified .env file.
# - Provide a singleton `settings` object for easy access throughout the application.

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    Defines the application's configuration settings.
    Pydantic automatically reads values from environment variables or the .env file.
    """
    # --- General Application Settings ---
    app_env: str = "dev"
    port: int = 8000

    # --- Vector Store Configuration ---
    vector_backend: str = "weaviate"  # Vector store backend: faiss, chroma, or weaviate
    storage_dir: str = "./storage"  # Directory to store indices and other data (for faiss/chroma)
    
    # Weaviate settings
    weaviate_url: str = "http://localhost:8090"  # Weaviate server URL (mapped to 8090 in docker)
    weaviate_api_key: str = ""  # Optional API key for authentication
    weaviate_class_name: str = "ChatbotUit"  # Weaviate class name (PascalCase)
    weaviate_use_cloud: bool = False  # true = cloud, false = local
    weaviate_grpc_port: int = 50051  # gRPC port for Weaviate

    # --- AI Model Configuration ---
    emb_model: str = "intfloat/multilingual-e5-base" # The embedding model to use
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Optional reranker model
    
    # --- Reranking Configuration ---
    use_reranking: bool = True  # Enable reranking with cross-encoder
    rerank_batch_size: int = 16  # Batch size for reranking operations
    rerank_max_length: int = 512  # Maximum input length for reranker
    rerank_top_k: int = 20  # Number of initial candidates to retrieve for reranking
    vietnamese_rerank_model: str = ""  # Optional Vietnamese-specific model

    # --- ChromaDB Specific Settings ---
    chroma_persist_dir: str = "./storage/chroma" # Directory for ChromaDB persistence

    # --- OpenSearch Settings for BM25 ---
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "rag_documents"
    opensearch_username: str = "admin"
    opensearch_password: str = "admin"
    opensearch_use_ssl: bool = False
    opensearch_verify_certs: bool = False

    # --- Hybrid Search Settings ---
    use_hybrid_search: bool = True  # Enable hybrid (vector + BM25) search
    bm25_weight: float = 0.5       # Weight for BM25 scores in fusion
    vector_weight: float = 0.5      # Weight for vector scores in fusion
    rrf_rank_constant: int = 60     # RRF rank constant (typically 60)

    # --- LLM Configuration ---
    openai_api_key: str = ""  # OpenAI API key (or OpenRouter API key)
    openai_base_url: str = ""  # Custom base URL (e.g., https://openrouter.ai/api/v1)
    gemini_api_key: str = ""  # Google Gemini API key
    llm_provider: str = "openai"  # LLM provider: openai, gemini, or openrouter
    llm_model: str = "gpt-4o-mini"  # Model to use (for OpenRouter: anthropic/claude-3.5-sonnet, etc.)
    llm_temperature: float = 0.0  # Temperature for LLM generation
    llm_max_tokens: int = 2000  # Max tokens for LLM response

    # --- 3-Tier Extraction Strategy ---
    tier2_model_id: str = "x-ai/grok-4.1-fast:free"  # Tier 2: Entity extraction
    tier2_temperature: float = 0.1
    tier2_max_tokens: int = 2048
    tier3_model_id: str = "x-ai/grok-4.1-fast:free"  # Tier 3: Rule extraction
    tier3_temperature: float = 0.0
    tier3_max_tokens: int = 4096

    # --- Neo4j Graph Database ---
    neo4j_uri: str = "bolt://localhost:7687"  # Neo4j connection URI
    neo4j_user: str = "neo4j"  # Neo4j username
    neo4j_password: str = "password"  # Neo4j password

    # --- Logging Configuration ---
    log_level: str = "INFO"

    # --- OpenRouter / VLM Configuration (optional) ---
    openrouter_api_key: str = ""  # OpenRouter API key (if using OpenRouter)
    vlm_model: str = ""  # Vision Language Model (optional)

    # Pydantic model configuration to specify the source of the settings
    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra fields in .env that are not defined here
    )

# Create a singleton instance of the Settings class to be used across the application
settings = Settings()
# store/vector/weaviate_store.py
#
# Description:
# Weaviate client initialization and schema management for Vietnamese RAG system.
# This module provides a clean, simple interface to Weaviate vector database.

import logging
import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.init import Auth
from typing import Optional

logger = logging.getLogger(__name__)

# Collection name for document chunks - matches actual data collection
DOCUMENT_COLLECTION = "VietnameseDocumentV3"


def get_weaviate_client(
    url: str = "http://localhost:8090",
    api_key: Optional[str] = None,
    timeout: int = 30
) -> weaviate.WeaviateClient:
    """
    Create and return a Weaviate client instance.
    
    Args:
        url: Weaviate server URL
        api_key: Optional API key for authentication
        timeout: Connection timeout in seconds
        
    Returns:
        Connected Weaviate client
    """
    try:
        # Parse URL to extract host and port
        host = url.replace("http://", "").replace("https://", "")
        port = 8090  # Default port
        
        # Extract port if it's in the URL
        if ":" in host:
            host_parts = host.split(":")
            host = host_parts[0]
            port = int(host_parts[1])
        
        # Determine if we're using HTTPS
        use_https = url.startswith("https://")
        
        # Create client configuration
        if api_key and api_key.strip():  # Check for non-empty API key
            auth_config = Auth.api_key(api_key)
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=use_https,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=use_https,
                auth_credentials=auth_config,
                skip_init_checks=False
            )
        else:
            # Connect without authentication (for self-hosted development)
            client = weaviate.connect_to_custom(
                http_host=host,
                http_port=port,
                http_secure=use_https,
                grpc_host=host,
                grpc_port=50051,
                grpc_secure=use_https,
                skip_init_checks=False
            )
        
        logger.info(f"Successfully connected to Weaviate at {url}")
        return client
        
    except Exception as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise


def create_document_collection(client: weaviate.WeaviateClient) -> bool:
    """
    Create the Vietnamese document collection schema in Weaviate.
    
    This schema is optimized for Vietnamese text with proper metadata fields
    for academic documents (syllabus, regulations, etc.).
    
    Args:
        client: Connected Weaviate client
        
    Returns:
        True if collection was created or already exists
    """
    try:
        # Check if collection already exists
        if client.collections.exists(DOCUMENT_COLLECTION):
            logger.info(f"Collection '{DOCUMENT_COLLECTION}' already exists")
            return True
        
        # Create collection with schema
        client.collections.create(
            name=DOCUMENT_COLLECTION,
            description="Vietnamese academic documents for RAG system",
            
            # Vectorizer configuration - we'll provide our own vectors
            vectorizer_config=Configure.Vectorizer.none(),
            
            # Vector index configuration - explicitly set distance metric
            vector_index_config=Configure.VectorIndex.hnsw(
                distance_metric=VectorDistances.COSINE
            ),
            
            # Properties/fields for document chunks
            properties=[
                Property(
                    name="text",
                    data_type=DataType.TEXT,
                    description="Document chunk text content",
                    skip_vectorization=True  # We handle vectorization ourselves
                ),
                Property(
                    name="doc_id",
                    data_type=DataType.TEXT,
                    description="Document identifier"
                ),
                Property(
                    name="chunk_id",
                    data_type=DataType.TEXT,
                    description="Chunk identifier within document"
                ),
                Property(
                    name="chunk_index",
                    data_type=DataType.INT,
                    description="Index of chunk in document"
                ),
                Property(
                    name="title",
                    data_type=DataType.TEXT,
                    description="Document title"
                ),
                Property(
                    name="page",
                    data_type=DataType.INT,
                    description="Page number in original document"
                ),
                Property(
                    name="doc_type",
                    data_type=DataType.TEXT,
                    description="Document type (syllabus, regulation, etc.)"
                ),
                Property(
                    name="faculty",
                    data_type=DataType.TEXT,
                    description="Faculty/department (CNTT, KHTN, etc.)"
                ),
                Property(
                    name="year",
                    data_type=DataType.INT,
                    description="Academic year"
                ),
                Property(
                    name="subject",
                    data_type=DataType.TEXT,
                    description="Subject or course code"
                ),
                Property(
                    name="section",
                    data_type=DataType.TEXT,
                    description="Document section"
                ),
                Property(
                    name="subsection",
                    data_type=DataType.TEXT,
                    description="Document subsection"
                ),
                Property(
                    name="language",
                    data_type=DataType.TEXT,
                    description="Document language (vi, en)"
                ),
                Property(
                    name="chapter",
                    data_type=DataType.TEXT,
                    description="Chapter (Chương) for legal documents"
                ),
                Property(
                    name="article",
                    data_type=DataType.TEXT,
                    description="Article (Điều) for legal documents"
                ),
                Property(
                    name="article_number",
                    data_type=DataType.INT,
                    description="Article number (numeric part of Điều)"
                ),
                Property(
                    name="structure_type",
                    data_type=DataType.TEXT,
                    description="Type: chapter, article, clause, point"
                ),
                Property(
                    name="filename",
                    data_type=DataType.TEXT,
                    description="Original PDF filename"
                ),
                Property(
                    name="metadata_json",
                    data_type=DataType.TEXT,
                    description="Additional metadata as JSON string"
                )
            ]
        )
        
        logger.info(f"Created collection '{DOCUMENT_COLLECTION}' successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create collection: {e}")
        return False


def delete_document_collection(client: weaviate.WeaviateClient) -> bool:
    """
    Delete the document collection (useful for testing/reset).
    
    Args:
        client: Connected Weaviate client
        
    Returns:
        True if deleted successfully
    """
    try:
        if client.collections.exists(DOCUMENT_COLLECTION):
            client.collections.delete(DOCUMENT_COLLECTION)
            logger.info(f"Deleted collection '{DOCUMENT_COLLECTION}'")
            return True
        else:
            logger.warning(f"Collection '{DOCUMENT_COLLECTION}' does not exist")
            return False
            
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return False


def ensure_collection_exists(client: weaviate.WeaviateClient) -> bool:
    """
    Ensure the document collection exists, create if not.
    
    Args:
        client: Connected Weaviate client
        
    Returns:
        True if collection exists or was created
    """
    if not client.collections.exists(DOCUMENT_COLLECTION):
        return create_document_collection(client)
    return True

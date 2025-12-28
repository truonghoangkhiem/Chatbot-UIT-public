# infrastructure/container.py
#
# Description:
# Dependency injection container for the RAG system.
# 
# This is the COMPOSITION ROOT of the application - the only place where
# we wire together all the dependencies. It belongs in the infrastructure
# layer because it knows about concrete implementations.

import logging
from typing import Optional

from core.domain.search_service import SearchService
from core.ports.repositories import VectorSearchRepository, KeywordSearchRepository
from core.ports.services import RerankingService, FusionService

logger = logging.getLogger(__name__)


# Lazy imports to avoid circular dependencies and maintain clean architecture
# These are infrastructure concerns and should only be imported when needed
def _lazy_import_adapters():
    """Lazy import adapters to maintain dependency direction."""
    from adapters.weaviate_vector_adapter import WeaviateVectorAdapter
    from adapters.opensearch_keyword_adapter import OpenSearchKeywordAdapter
    from adapters.service_adapters import HybridFusionAdapter
    from adapters.cross_encoder_reranker import create_reranking_service
    from sentence_transformers import SentenceTransformer
    from app.config.settings import settings
    
    return {
        'WeaviateVectorAdapter': WeaviateVectorAdapter,
        'OpenSearchKeywordAdapter': OpenSearchKeywordAdapter,
        'HybridFusionAdapter': HybridFusionAdapter,
        'create_reranking_service': create_reranking_service,
        'SentenceTransformer': SentenceTransformer,
        'settings': settings
    }


class DIContainer:
    """
    Dependency Injection Container for the RAG system.
    
    This is the COMPOSITION ROOT - the place where the application is wired together.
    It follows the Ports & Adapters pattern by:
    1. Creating concrete implementations (adapters) for the ports
    2. Injecting dependencies into the core domain services
    3. Managing the lifecycle of all components
    
    Note: It's perfectly acceptable for this class to import and know about
    adapters because that's its job - to compose the application.
    """
    
    def __init__(self):
        self._vector_repository: Optional[VectorSearchRepository] = None
        self._keyword_repository: Optional[KeywordSearchRepository] = None
        self._reranking_service: Optional[RerankingService] = None
        self._fusion_service: Optional[FusionService] = None
        self._search_service: Optional[SearchService] = None
    
    def get_vector_repository(self) -> VectorSearchRepository:
        """Get or create the vector search repository."""
        if self._vector_repository is None:
            deps = _lazy_import_adapters()
            WeaviateVectorAdapter = deps['WeaviateVectorAdapter']
            SentenceTransformer = deps['SentenceTransformer']
            settings = deps['settings']
            
            # Create a simple embedding wrapper that matches Weaviate adapter interface
            class SimpleEmbedding:
                def __init__(self, model_name: str):
                    self.model = SentenceTransformer(model_name)
                
                def get_text_embedding(self, text: str):
                    return self.model.encode(text).tolist()
            
            embedding_model = SimpleEmbedding(settings.emb_model)
            
            self._vector_repository = WeaviateVectorAdapter(
                weaviate_url=settings.weaviate_url,
                embedding_model=embedding_model,
                api_key=settings.weaviate_api_key if settings.weaviate_api_key else None
            )
            logger.info("Created Weaviate vector repository")
        
        return self._vector_repository
    
    def get_keyword_repository(self) -> Optional[KeywordSearchRepository]:
        """Get or create the keyword search repository."""
        deps = _lazy_import_adapters()
        settings = deps['settings']
        
        if not settings.use_hybrid_search:
            return None
        
        if self._keyword_repository is None:
            try:
                from infrastructure.store.opensearch.client import get_opensearch_client
                OpenSearchKeywordAdapter = deps['OpenSearchKeywordAdapter']
                opensearch_client = get_opensearch_client()
                self._keyword_repository = OpenSearchKeywordAdapter(opensearch_client)
                logger.info("Created OpenSearch keyword repository")
            except Exception as e:
                logger.warning(f"Could not create OpenSearch keyword repository: {e}")
                return None
        
        return self._keyword_repository
    
    def get_reranking_service(self) -> Optional[RerankingService]:
        """Get or create the reranking service."""
        deps = _lazy_import_adapters()
        settings = deps['settings']
        
        if not settings.use_reranking or not settings.rerank_model:
            return None
        
        if self._reranking_service is None:
            try:
                create_reranking_service = deps['create_reranking_service']
                # Use our new enhanced cross-encoder reranking service
                self._reranking_service = create_reranking_service(
                    model_name=settings.rerank_model,
                    multilingual=True,
                    vietnamese_model_name=settings.vietnamese_rerank_model if settings.vietnamese_rerank_model else None,
                    batch_size=settings.rerank_batch_size,
                    max_length=settings.rerank_max_length
                )
                logger.info(f"Created enhanced CrossEncoder reranking service: {settings.rerank_model}")
                
                # Log reranking configuration
                model_info = self._reranking_service.get_model_info()
                logger.info(f"Reranking service configuration: {model_info}")
                
            except Exception as e:
                logger.warning(f"Could not create reranking service: {e}")
                return None
        
        return self._reranking_service
    
    def get_fusion_service(self) -> Optional[FusionService]:
        """Get or create the fusion service."""
        deps = _lazy_import_adapters()
        settings = deps['settings']
        
        if not settings.use_hybrid_search:
            return None
        
        if self._fusion_service is None:
            HybridFusionAdapter = deps['HybridFusionAdapter']
            self._fusion_service = HybridFusionAdapter(
                rrf_constant=settings.rrf_rank_constant
            )
            logger.info("Created hybrid fusion service")
        
        return self._fusion_service
    
    def get_search_service(self) -> SearchService:
        """Get or create the main search service."""
        if self._search_service is None:
            vector_repo = self.get_vector_repository()
            keyword_repo = self.get_keyword_repository()
            rerank_service = self.get_reranking_service()
            fusion_service = self.get_fusion_service()
            
            self._search_service = SearchService(
                vector_repository=vector_repo,
                keyword_repository=keyword_repo,
                reranking_service=rerank_service,
                fusion_service=fusion_service,
                highlighting_service=None  # Not implemented yet
            )
            
            logger.info("Created core search service with injected dependencies")
        
        return self._search_service
    
    def reset(self):
        """Reset all cached instances (useful for testing)."""
        self._vector_repository = None
        self._keyword_repository = None
        self._reranking_service = None
        self._fusion_service = None
        self._search_service = None
        logger.info("Reset DI container")


# Global container instance
_container = DIContainer()


def get_container() -> DIContainer:
    """Get the global DI container instance."""
    return _container


# Convenience functions for direct access
def get_search_service() -> SearchService:
    """Get the configured search service."""
    return _container.get_search_service()


def reset_container():
    """Reset the global container (mainly for testing)."""
    _container.reset()

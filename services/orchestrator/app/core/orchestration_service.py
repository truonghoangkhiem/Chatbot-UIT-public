"""
Core orchestrator service.

This service coordinates between RAG retrieval and agent generation,
implementing the main business logic of the orchestration pipeline.

Enhanced with IRCoT (Interleaving Retrieval with Chain-of-Thought) support
for complex multi-hop questions.
"""

import uuid
import time
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from ..core.domain import (
    OrchestrationRequest,
    OrchestrationResponse,
    AgentRequest,
    RAGContext,
    ConversationMessage,
    ConversationRole,
    MessageType
)
from ..ports.agent_ports import AgentPort, RAGServicePort, ConversationManagerPort
from ..core.context_domain_service import ContextDomainService
from ..core.exceptions import AgentProcessingFailedException, RAGRetrievalFailedException
from ..core.ircot_config import IRCoTConfig, IRCoTMode
from ..core.ircot_service import IRCoTReasoningService

logger = logging.getLogger(__name__)


class OrchestrationService:
    """
    Main orchestration service that coordinates RAG and Agent services.
    
    This service implements the core business logic for the orchestration
    pipeline, coordinating between document retrieval and response generation.
    
    Enhanced with IRCoT (Interleaving Retrieval with Chain-of-Thought) support
    for complex multi-hop questions that require iterative reasoning.
    """
    
    def __init__(
        self,
        agent_port: AgentPort,
        rag_port: RAGServicePort,
        conversation_manager: ConversationManagerPort,
        default_system_prompt: Optional[str] = None,
        ircot_config: Optional[IRCoTConfig] = None
    ):
        """
        Initialize the orchestration service.
        
        Args:
            agent_port: Port for agent communication
            rag_port: Port for RAG service communication
            conversation_manager: Port for conversation management
            default_system_prompt: Default system prompt to use
            ircot_config: Optional IRCoT configuration for complex queries
        """
        self.agent_port = agent_port
        self.rag_port = rag_port
        self.conversation_manager = conversation_manager
        self.default_system_prompt = default_system_prompt or self._get_default_system_prompt()
        
        # Initialize IRCoT service
        self.ircot_config = ircot_config or IRCoTConfig()
        self.ircot_service = IRCoTReasoningService(
            agent_port=agent_port,
            rag_port=rag_port,
            config=self.ircot_config
        )
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for the agent."""
        return """Bạn là một trợ lý AI thông minh và hữu ích cho Chatbot-UIT. Nhiệm vụ của bạn là:

1. Trả lời câu hỏi dựa trên thông tin được cung cấp từ hệ thống tìm kiếm
2. Nếu thông tin không đủ để trả lời, hãy thành thật nói rằng bạn không có đủ thông tin
3. Luôn trả lời bằng tiếng Việt một cách tự nhiên và dễ hiểu
4. Cung cấp thông tin chính xác và hữu ích cho người dùng
5. Nếu cần, hãy yêu cầu người dùng cung cấp thêm thông tin để trả lời tốt hơn

Hãy trả lời một cách thân thiện và chuyên nghiệp."""
    
    async def process_request(self, request: OrchestrationRequest) -> OrchestrationResponse:
        """
        Process a complete orchestration request.
        
        Args:
            request: The orchestration request
            
        Returns:
            OrchestrationResponse containing the generated response and metadata
        """
        start_time = time.time()
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        # Get or create conversation context
        context = await self._get_or_create_context(session_id, request.conversation_context)
        
        # Add user message to context
        user_message = ConversationMessage(
            role=ConversationRole.USER,
            content=request.user_query,
            timestamp=datetime.now(),
            message_type=MessageType.TEXT
        )
        context.add_message(user_message)
        
        rag_context = None
        processing_stats = {}
        
        # Retrieve context using RAG if enabled
        if request.use_rag:
            try:
                rag_start = time.time()
                rag_data = await self.rag_port.retrieve_context(
                    query=request.user_query,
                    top_k=request.rag_top_k
                )
                rag_end = time.time()
                
                rag_context = RAGContext(
                    query=request.user_query,
                    retrieved_documents=rag_data.get("retrieved_documents", []),
                    search_metadata=rag_data.get("search_metadata"),
                    relevance_scores=rag_data.get("relevance_scores", [])
                )
                
                processing_stats["rag_time"] = rag_end - rag_start
                processing_stats["documents_retrieved"] = len(rag_context.retrieved_documents)
                
            except Exception as e:
                # For non-critical RAG failures, continue without RAG context
                # Critical failures should be handled by caller
                processing_stats["rag_error"] = str(e)
                rag_context = None
        
        # Prepare agent request
        agent_request = self._prepare_agent_request(
            user_query=request.user_query,
            rag_context=rag_context,
            context=context,
            model=request.agent_model,
            metadata=request.metadata
        )
        
        # Generate response using agent
        try:
            agent_start = time.time()
            agent_response = await self.agent_port.generate_response(agent_request)
            agent_end = time.time()
            
            processing_stats["agent_time"] = agent_end - agent_start
            processing_stats["tokens_used"] = agent_response.tokens_used
            
            agent_response_content = agent_response.content
            agent_metadata = agent_response.metadata
            
        except Exception as e:
            # Raise domain exception - let presentation layer handle user messages
            raise AgentProcessingFailedException(
                agent_error=str(e),
                details={
                    "session_id": session_id,
                    "user_query": request.user_query,
                    "processing_stats": processing_stats
                },
                cause=e
            )
        
        # Add assistant message to context
        assistant_message = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=agent_response_content,
            timestamp=datetime.now(),
            message_type=MessageType.TEXT
        )
        context.add_message(assistant_message)
        
        # Update conversation context
        await self.conversation_manager.update_context(context)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        processing_stats["total_time"] = total_time
        
        return OrchestrationResponse(
            response=agent_response_content,
            session_id=session_id,
            rag_context=rag_context,
            agent_metadata=agent_metadata,
            processing_stats=processing_stats,
            timestamp=datetime.now()
        )
    
    async def _get_or_create_context(
        self, 
        session_id: str, 
        provided_context: Optional[Any] = None
    ) -> Any:
        """
        Get existing or create new conversation context.
        
        Args:
            session_id: Session identifier
            provided_context: Optionally provided context
            
        Returns:
            ConversationContext instance
        """
        # Try to get existing context
        context = await self.conversation_manager.get_context(session_id)
        
        if context is None:
            # Create new context
            context = await self.conversation_manager.create_context(
                session_id=session_id,
                system_prompt=self.default_system_prompt
            )
        
        return context
    
    def _prepare_agent_request(
        self,
        user_query: str,
        rag_context: Optional[RAGContext],
        context: Any,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentRequest:
        """
        Prepare the agent request with RAG context if available.
        
        Args:
            user_query: The user's query
            rag_context: Retrieved context from RAG system
            context: Conversation context
            model: Optional model specification
            metadata: Optional request metadata
            
        Returns:
            AgentRequest ready for agent processing
        """
        # Prepare the prompt with RAG context if available
        enhanced_prompt = user_query
        context_data = None
        
        if rag_context and rag_context.retrieved_documents:
            # Extract context data using domain service (no formatting)
            context_data = self._extract_context_data(rag_context)
            
            # Store context data in conversation context for agent to use
            if context:
                context.metadata = context.metadata or {}
                context.metadata["rag_context"] = context_data
        
        return AgentRequest(
            prompt=enhanced_prompt,
            context=context,
            model=model,
            max_tokens=metadata.get("max_tokens") if metadata else None,
            temperature=metadata.get("temperature") if metadata else None,
            stream=False,
            metadata=metadata
        )
    
    def _extract_context_data(self, rag_context: RAGContext) -> Dict[str, Any]:
        """
        Extract relevant context data (pure domain logic).
        
        Args:
            rag_context: RAG context containing retrieved documents
            
        Returns:
            Processed context data for agent consumption
        """
        context_service = ContextDomainService()
        
        # Extract relevant documents using domain service
        relevant_docs = context_service.extract_relevant_documents(rag_context)
        
        # Assess context quality
        quality_assessment = context_service.assess_context_quality(relevant_docs)
        
        return {
            "documents": relevant_docs,
            "quality": quality_assessment,
            "total_documents": len(rag_context.retrieved_documents),
            "search_metadata": rag_context.search_metadata or {}
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all components.
        
        Returns:
            Dictionary containing health status of all components
        """
        health_status = {
            "orchestrator": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ircot_enabled": self.ircot_config.enabled
        }
        
        # Check agent service
        try:
            agent_healthy = await self.agent_port.validate_connection()
            health_status["agent"] = "healthy" if agent_healthy else "unhealthy"
        except Exception as e:
            health_status["agent"] = f"error: {str(e)}"
        
        # Check RAG service
        try:
            rag_healthy = await self.rag_port.health_check()
            health_status["rag"] = "healthy" if rag_healthy else "unhealthy"
        except Exception as e:
            health_status["rag"] = f"error: {str(e)}"
        
        # Check conversation manager (always healthy for in-memory implementation)
        health_status["conversation_manager"] = "healthy"
        
        # Overall status
        all_healthy = all(
            status == "healthy" 
            for key, status in health_status.items() 
            if key not in ["timestamp", "orchestrator", "ircot_enabled"]
        )
        health_status["overall"] = "healthy" if all_healthy else "degraded"
        
        return health_status
    
    async def process_request_with_ircot(
        self,
        request: OrchestrationRequest,
        complexity: str = "complex",
        complexity_score: float = 7.0
    ) -> OrchestrationResponse:
        """
        Process request using IRCoT (Interleaving Retrieval with Chain-of-Thought).
        
        This method should be called when SmartPlannerAgent determines that
        the query is complex and requires multi-hop reasoning.
        
        IRCoT Algorithm:
        1. Initial retrieval based on original query
        2. Generate CoT reasoning step
        3. If more info needed, generate new search query from reasoning
        4. Retrieve additional context with new query
        5. Repeat until confident or max iterations reached
        6. Use accumulated context for final answer generation
        
        Args:
            request: The orchestration request
            complexity: Complexity level ("simple", "medium", "complex")
            complexity_score: Numeric complexity score (0-10)
            
        Returns:
            OrchestrationResponse with IRCoT-enhanced context
        """
        start_time = time.time()
        session_id = request.session_id or str(uuid.uuid4())
        processing_stats = {
            "ircot_mode": True,
            "complexity": complexity,
            "complexity_score": complexity_score
        }
        
        # Check if IRCoT should be used
        use_ircot = self.ircot_config.should_use_ircot(complexity, complexity_score)
        
        if not use_ircot:
            # Fall back to standard processing
            logger.info(f"IRCoT not triggered for query (complexity={complexity})")
            processing_stats["ircot_mode"] = False
            return await self.process_request(request)
        
        logger.info(f"🔄 Processing with IRCoT: {request.user_query[:50]}...")
        
        # Get or create conversation context
        context = await self._get_or_create_context(session_id, request.conversation_context)
        
        # Add user message to context
        user_message = ConversationMessage(
            role=ConversationRole.USER,
            content=request.user_query,
            timestamp=datetime.now(),
            message_type=MessageType.TEXT
        )
        context.add_message(user_message)
        
        rag_context = None
        
        try:
            # Step 1: Execute IRCoT reasoning with dynamic retrieval
            ircot_start = time.time()
            ircot_result = await self.ircot_service.reason_with_retrieval(
                query=request.user_query,
                initial_context=None,  # Let IRCoT handle initial retrieval
                extracted_filters=None
            )
            ircot_end = time.time()
            
            processing_stats["ircot_time"] = ircot_end - ircot_start
            processing_stats["ircot_iterations"] = ircot_result.total_iterations
            processing_stats["ircot_early_stopped"] = ircot_result.early_stopped
            processing_stats["ircot_confidence"] = ircot_result.final_confidence
            processing_stats["ircot_documents_accumulated"] = len(ircot_result.accumulated_context)
            
            # Build RAG context from IRCoT accumulated context
            rag_context = RAGContext(
                query=request.user_query,
                retrieved_documents=ircot_result.accumulated_context,
                search_metadata={
                    "ircot_iterations": ircot_result.total_iterations,
                    "ircot_queries": ircot_result.get_all_search_queries(),
                    "ircot_reasoning": ircot_result.final_reasoning
                },
                relevance_scores=[
                    doc.get("score", 0.0) 
                    for doc in ircot_result.accumulated_context
                ]
            )
            
            processing_stats["documents_retrieved"] = len(rag_context.retrieved_documents)
            
        except Exception as e:
            logger.error(f"IRCoT processing error: {e}")
            processing_stats["ircot_error"] = str(e)
            
            # Fallback to standard RAG
            try:
                rag_data = await self.rag_port.retrieve_context(
                    query=request.user_query,
                    top_k=request.rag_top_k
                )
                rag_context = RAGContext(
                    query=request.user_query,
                    retrieved_documents=rag_data.get("retrieved_documents", []),
                    search_metadata=rag_data.get("search_metadata"),
                    relevance_scores=rag_data.get("relevance_scores", [])
                )
            except Exception as fallback_error:
                processing_stats["rag_fallback_error"] = str(fallback_error)
        
        # Prepare agent request with IRCoT-enhanced context
        agent_request = self._prepare_agent_request_with_ircot(
            user_query=request.user_query,
            rag_context=rag_context,
            ircot_reasoning=ircot_result.final_reasoning if 'ircot_result' in locals() else None,
            context=context,
            model=request.agent_model,
            metadata=request.metadata
        )
        
        # Generate final response
        try:
            agent_start = time.time()
            agent_response = await self.agent_port.generate_response(agent_request)
            agent_end = time.time()
            
            processing_stats["agent_time"] = agent_end - agent_start
            processing_stats["tokens_used"] = agent_response.tokens_used
            
            agent_response_content = agent_response.content
            agent_metadata = agent_response.metadata or {}
            
            # Add IRCoT info to metadata
            if 'ircot_result' in locals():
                agent_metadata["ircot"] = ircot_result.to_dict()
            
        except Exception as e:
            raise AgentProcessingFailedException(
                agent_error=str(e),
                details={
                    "session_id": session_id,
                    "user_query": request.user_query,
                    "ircot_mode": True,
                    "processing_stats": processing_stats
                },
                cause=e
            )
        
        # Add assistant message to context
        assistant_message = ConversationMessage(
            role=ConversationRole.ASSISTANT,
            content=agent_response_content,
            timestamp=datetime.now(),
            message_type=MessageType.TEXT
        )
        context.add_message(assistant_message)
        
        # Update conversation context
        await self.conversation_manager.update_context(context)
        
        # Calculate total processing time
        total_time = time.time() - start_time
        processing_stats["total_time"] = total_time
        
        return OrchestrationResponse(
            response=agent_response_content,
            session_id=session_id,
            rag_context=rag_context,
            agent_metadata=agent_metadata,
            processing_stats=processing_stats,
            timestamp=datetime.now()
        )
    
    def _prepare_agent_request_with_ircot(
        self,
        user_query: str,
        rag_context: Optional[RAGContext],
        ircot_reasoning: Optional[str],
        context: Any,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentRequest:
        """
        Prepare agent request with IRCoT reasoning context.
        
        This enhances the standard request preparation by including
        the chain-of-thought reasoning from IRCoT iterations.
        
        Args:
            user_query: The user's query
            rag_context: Retrieved context from IRCoT
            ircot_reasoning: Compiled reasoning from IRCoT iterations
            context: Conversation context
            model: Optional model specification
            metadata: Optional request metadata
            
        Returns:
            AgentRequest ready for agent processing
        """
        # Build enhanced prompt with IRCoT reasoning
        enhanced_prompt = user_query
        context_data = None
        
        if rag_context and rag_context.retrieved_documents:
            # Extract context data
            context_data = self._extract_context_data(rag_context)
            
            # Add IRCoT reasoning to context
            if ircot_reasoning:
                context_data["ircot_reasoning"] = ircot_reasoning
            
            # Store in conversation context
            if context:
                context.metadata = context.metadata or {}
                context.metadata["rag_context"] = context_data
                context.metadata["ircot_mode"] = True
        
        return AgentRequest(
            prompt=enhanced_prompt,
            context=context,
            model=model,
            max_tokens=metadata.get("max_tokens") if metadata else None,
            temperature=metadata.get("temperature") if metadata else None,
            stream=False,
            metadata={
                **(metadata or {}),
                "ircot_mode": True,
                "has_ircot_reasoning": ircot_reasoning is not None
            }
        )
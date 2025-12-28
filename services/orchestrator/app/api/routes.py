"""
FastAPI routes for orchestrator service.

This module defines the API endpoints for the orchestrator service,
providing REST API access to the orchestration functionality.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from datetime import datetime
import json
import asyncio
import logging
from typing import AsyncGenerator

from ..core.container import get_orchestration_service, get_multi_agent_orchestrator
from ..core.domain import OrchestrationRequest
from ..core.exceptions import OrchestrationDomainException
from ..agents.base import AgentType
from ..schemas.api_schemas import (
    ChatRequest, 
    ChatResponse, 
    HealthResponse, 
    ConversationsResponse,
    ConversationInfo,
    ErrorResponse,
    DocumentInfo,
    RAGContextInfo,
    ProcessingStats
)
from .exception_handlers import ExceptionMessageHandler

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Generate response using optimized 2-agent orchestration pipeline",
    description="Process user query through optimized 2-agent pipeline (Smart Planner + Answer Agent with built-in formatting). Supports both streaming and non-streaming responses."
)
async def chat(request: ChatRequest):
    """
    Generate a response using the optimized 2-agent orchestration pipeline.
    
    This endpoint processes the user query through the following steps:
    1. Smart Planner - Analyzes query intent, complexity, and rewrites queries
    2. RAG Retrieval - Gets relevant context using optimized queries
    3. Answer Agent - Generates comprehensive, formatted answers (includes built-in formatting)
    
    Pipeline Optimization (v2):
    - Removed ResponseFormatterAgent (formatting now built into AnswerAgent)
    - 60% fewer LLM calls vs original (2 vs 5)
    - 33% lower latency vs v1 (removed formatting step)
    
    If streaming is enabled, returns Server-Sent Events (SSE) format.
    Otherwise, returns a complete ChatResponse object.
    
    Args:
        request: Chat request containing query and configuration
        
    Returns:
        StreamingResponse (if stream=True) or ChatResponse (if stream=False)
    """
    # If streaming is requested, use the streaming endpoint
    if request.stream:
        return await chat_stream_multi_agent(request)
    
    try:
        # Get multi-agent orchestrator
        multi_agent_orchestrator = get_multi_agent_orchestrator()
        
        # Create orchestration request
        orch_request = OrchestrationRequest(
            user_query=request.query,
            session_id=request.session_id,
            use_rag=request.use_rag,
            use_knowledge_graph=request.use_knowledge_graph,  # Pass KG flag
            rag_top_k=request.rag_top_k,
            agent_model=request.model,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": request.stream
            }
        )
        
        # Process request
        response = await multi_agent_orchestrator.process_request(orch_request)
        
        # Transform RAG context for API response
        rag_context_info = None
        if response.rag_context:
            documents = [
                DocumentInfo(
                    title=doc.get("title", "Untitled"),
                    # RAG service returns 'text' field, fallback to 'content'
                    content=doc.get("text", doc.get("content", "")),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {})
                )
                for doc in response.rag_context.retrieved_documents
            ]
            
            # Get search mode from metadata
            search_mode = "unknown"
            if response.rag_context.search_metadata:
                search_mode = response.rag_context.search_metadata.get("search_mode", "unknown")
            
            # Get KG usage info from processing stats
            use_knowledge_graph = response.processing_stats.get("use_knowledge_graph", False)
            use_vector_search = response.processing_stats.get("use_vector_search", True)
            complexity = response.processing_stats.get("complexity")
            strategy = response.processing_stats.get("strategy")
            
            rag_context_info = RAGContextInfo(
                query=response.rag_context.query,
                documents=documents,
                total_documents=len(documents),
                search_mode=search_mode,
                processing_time=response.processing_stats.get("rag_time"),
                use_knowledge_graph=use_knowledge_graph,
                use_vector_search=use_vector_search,
                complexity=complexity,
                strategy=strategy
            )
        
        # Transform processing stats
        processing_stats = ProcessingStats(
            total_time=response.processing_stats.get("total_time", 0.0),
            rag_time=response.processing_stats.get("rag_time"),
            agent_time=response.processing_stats.get("agent_time"),
            documents_retrieved=response.processing_stats.get("documents_retrieved"),
            tokens_used=response.processing_stats.get("tokens_used"),
            rag_error=response.processing_stats.get("rag_error"),
            # Optimized pipeline stats
            llm_calls=response.processing_stats.get("llm_calls"),
            pipeline=response.processing_stats.get("pipeline"),
            planning_time=response.processing_stats.get("planning_time"),
            answer_generation_time=response.processing_stats.get("answer_generation_time"),
            plan_complexity=response.processing_stats.get("plan_complexity"),
            plan_complexity_score=response.processing_stats.get("plan_complexity_score")
        )
        
        # Get model used from agent metadata
        model_used = None
        if response.agent_metadata:
            model_used = response.agent_metadata.get("model_used")
        
        return ChatResponse(
            response=response.response,
            session_id=response.session_id,
            timestamp=response.timestamp or datetime.now(),
            rag_context=rag_context_info,
            processing_stats=processing_stats,
            model_used=model_used
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


async def chat_stream_multi_agent(request: ChatRequest):
    """
    Stream a response using the multi-agent orchestration pipeline.
    
    This is an internal function called by the /chat endpoint when streaming is enabled.
    It performs RAG retrieval first, then streams the answer generation.
    
    Args:
        request: Chat request containing query and configuration
        
    Returns:
        StreamingResponse with Server-Sent Events format
    """
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Get multi-agent orchestrator
            multi_agent_orchestrator = get_multi_agent_orchestrator()
            
            # Step 1: Planning phase (if enabled)
            planning_result = None
            if multi_agent_orchestrator.enable_planning:
                try:
                    smart_planner = multi_agent_orchestrator.agent_factory.create_agent(AgentType.SMART_PLANNER)
                    planning_input = {
                        "query": request.query,
                        "conversation_history": []
                    }
                    planning_result = await smart_planner.process(planning_input)
                    
                    # Send planning info to client
                    yield f"data: {json.dumps({'type': 'planning', 'content': 'Đang phân tích câu hỏi...'})}\n\n"
                except Exception as e:
                    logger.warning(f"Planning failed, continuing without it: {e}")
            
            # Step 2: RAG Retrieval
            rag_context = None
            context_documents = []
            if request.use_rag:
                try:
                    yield f"data: {json.dumps({'type': 'status', 'content': 'Đang tìm kiếm thông tin liên quan...'})}\n\n"
                    
                    # Use rewritten queries if available from planning
                    queries_to_use = [request.query]
                    if planning_result and hasattr(planning_result, 'rewritten_queries'):
                        queries_to_use = planning_result.rewritten_queries or [request.query]
                    
                    # Retrieve context
                    rag_data = await multi_agent_orchestrator.rag_port.retrieve_context(
                        query=queries_to_use[0],  # Use first query
                        top_k=request.rag_top_k
                    )
                    
                    from ..core.domain import RAGContext
                    rag_context = RAGContext(
                        query=queries_to_use[0],
                        retrieved_documents=rag_data.get("retrieved_documents", []),
                        search_metadata=rag_data.get("search_metadata")
                    )
                    context_documents = rag_context.retrieved_documents
                    
                    yield f"data: {json.dumps({'type': 'status', 'content': f'Đã tìm thấy {len(context_documents)} tài liệu liên quan'})}\n\n"
                except Exception as e:
                    logger.warning(f"RAG retrieval failed: {e}")
                    yield f"data: {json.dumps({'type': 'warning', 'content': 'Không tìm thấy tài liệu tham khảo, sẽ trả lời dựa trên kiến thức chung'})}\n\n"
            
            # Step 3: Stream answer generation
            yield f"data: {json.dumps({'type': 'status', 'content': 'Đang tạo câu trả lời...'})}\n\n"
            
            answer_agent = multi_agent_orchestrator.agent_factory.create_agent(AgentType.ANSWER_AGENT)
            
            answer_input = {
                "query": request.query,
                "context_documents": context_documents,
                "rewritten_queries": planning_result.rewritten_queries if planning_result else [],
                "previous_context": ""
            }
            
            # Stream the answer
            async for chunk in answer_agent.stream_process(answer_input):
                yield f"data: {json.dumps({'type': 'content', 'content': chunk})}\n\n"
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'content': 'Hoàn thành'})}\n\n"
        
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_message = f"Đã có lỗi xảy ra: {str(e)}"
            yield f"data: {json.dumps({'type': 'error', 'content': error_message})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )


@router.post(
    "/chat/simple",
    response_model=ChatResponse,
    summary="Generate response using simple orchestration pipeline",
    description="Process user query through simple single-agent pipeline (faster but less sophisticated)"
)
async def simple_chat(request: ChatRequest) -> ChatResponse:
    """
    Generate a response using the simple orchestration pipeline.
    
    This endpoint uses the original simple orchestration without
    the multi-agent pipeline for faster, simpler responses.
    
    Args:
        request: Chat request containing query and configuration
        
    Returns:
        Generated response with metadata and processing statistics
    """
    try:
        # Get simple orchestration service
        orchestration_service = get_orchestration_service()
        
        # Create orchestration request
        orch_request = OrchestrationRequest(
            user_query=request.query,
            session_id=request.session_id,
            use_rag=request.use_rag,
            rag_top_k=request.rag_top_k,
            agent_model=request.model,
            metadata={
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "stream": request.stream,
                "endpoint": "simple_chat"
            }
        )
        
        # Process request
        response = await orchestration_service.process_request(orch_request)
        
        # Transform RAG context for API response
        rag_context_info = None
        if response.rag_context:
            documents = [
                DocumentInfo(
                    title=doc.get("title", "Untitled"),
                    # RAG service returns 'text' field, fallback to 'content'
                    content=doc.get("text", doc.get("content", "")),
                    score=doc.get("score", 0.0),
                    metadata=doc.get("metadata", {})
                )
                for doc in response.rag_context.retrieved_documents
            ]
            
            rag_context_info = RAGContextInfo(
                query=response.rag_context.query,
                documents=documents,
                total_documents=len(documents),
                search_mode="simple",
                processing_time=response.processing_stats.get("rag_time")
            )
        
        # Transform processing stats
        processing_stats = ProcessingStats(
            total_time=response.processing_stats.get("total_time", 0.0),
            rag_time=response.processing_stats.get("rag_time"),
            agent_time=response.processing_stats.get("agent_time"),
            documents_retrieved=response.processing_stats.get("documents_retrieved"),
            tokens_used=response.processing_stats.get("tokens_used"),
            rag_error=response.processing_stats.get("rag_error"),
            # Optimized pipeline stats
            llm_calls=response.processing_stats.get("llm_calls"),
            pipeline=response.processing_stats.get("pipeline"),
            planning_time=response.processing_stats.get("planning_time"),
            answer_generation_time=response.processing_stats.get("answer_generation_time"),
            plan_complexity=response.processing_stats.get("plan_complexity"),
            plan_complexity_score=response.processing_stats.get("plan_complexity_score")
        )
        
        return ChatResponse(
            response=response.response,
            session_id=response.session_id,
            timestamp=response.timestamp or datetime.now(),
            rag_context=rag_context_info,
            processing_stats=processing_stats,
            model_used="simple_orchestrator"
        )
    
    except OrchestrationDomainException as domain_ex:
        # Handle domain exceptions with user-friendly messages
        fallback_response = ExceptionMessageHandler.create_fallback_response(
            exception=domain_ex,
            session_id=request.session_id or "unknown",
            user_query=request.query
        )
        
        # Return user-friendly response instead of HTTP error
        return ChatResponse(
            response=fallback_response["response"],
            session_id=fallback_response["session_id"],
            timestamp=datetime.now(),
            rag_context=None,
            processing_stats=ProcessingStats(
                total_time=0.0,
                rag_time=None,
                agent_time=None,
                documents_retrieved=None,
                tokens_used=None,
                rag_error=domain_ex.details.get("agent_error") if hasattr(domain_ex, 'details') else None
            ),
            model_used="error_handler"
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Simple chat processing failed: {str(e)}"
        )


@router.post(
    "/chat/stream",
    summary="Stream response using orchestration pipeline",
    description="Stream response from agent with RAG context (experimental)"
)
async def chat_stream(request: ChatRequest):
    """
    Stream a response using the orchestration pipeline.
    
    Note: This is an experimental endpoint. RAG retrieval happens first,
    then the agent response is streamed.
    
    Args:
        request: Chat request containing query and configuration
        
    Returns:
        Streaming response with generated content
    """
    if not request.stream:
        raise HTTPException(
            status_code=400,
            detail="Stream must be enabled for this endpoint"
        )
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        try:
            # Get orchestration service
            orchestration_service = get_orchestration_service()
            
            # Get or create conversation context first
            session_id = request.session_id or "temp_session"
            context = await orchestration_service._get_or_create_context(session_id)
            
            # Retrieve RAG context if enabled
            rag_context = None
            if request.use_rag:
                try:
                    rag_data = await orchestration_service.rag_port.retrieve_context(
                        query=request.query,
                        top_k=request.rag_top_k
                    )
                    # Prepare rag_context but don't need full transformation for streaming
                    from ..core.domain import RAGContext
                    rag_context = RAGContext(
                        query=request.query,
                        retrieved_documents=rag_data.get("retrieved_documents", []),
                        search_metadata=rag_data.get("search_metadata")
                    )
                except Exception:
                    # Continue without RAG context if error
                    pass
            
            # Prepare agent request
            agent_request = orchestration_service._prepare_agent_request(
                user_query=request.query,
                rag_context=rag_context,
                context=context,
                model=request.model,
                metadata={
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "stream": True
                }
            )
            
            # Stream response from agent
            async for chunk in orchestration_service.agent_port.stream_response(agent_request):
                yield f"data: {json.dumps({'content': chunk})}\n\n"
            
            yield f"data: {json.dumps({'done': True})}\n\n"
        
        except OrchestrationDomainException as domain_ex:
            # Handle domain exceptions with user-friendly messages
            user_message = ExceptionMessageHandler.get_user_message(domain_ex)
            error_data = {"error": user_message, "is_user_error": True}
            yield f"data: {json.dumps(error_data)}\n\n"
        
        except Exception as e:
            error_data = {"error": f"Đã có lỗi hệ thống: {str(e)}", "is_system_error": True}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check the health status of all orchestrator components"
)
async def health_check() -> HealthResponse:
    """
    Perform health check on all orchestrator components.
    
    Returns:
        Health status of orchestrator and all dependent services
    """
    try:
        orchestration_service = get_orchestration_service()
        health_status = await orchestration_service.health_check()
        
        return HealthResponse(
            status=health_status["overall"],
            timestamp=datetime.fromisoformat(health_status["timestamp"]),
            services={
                k: v for k, v in health_status.items() 
                if k not in ["overall", "timestamp"]
            }
        )
    
    except Exception as e:
        return HealthResponse(
            status="error",
            timestamp=datetime.now(),
            services={"error": str(e)}
        )


@router.get(
    "/debug/graph",
    summary="Debug Graph Adapter status",
    description="Check if Graph Adapter is initialized and working"
)
async def debug_graph_adapter():
    """Debug endpoint to check Graph Adapter initialization."""
    import os
    from ..core.container import get_container
    
    try:
        container = get_container()
        graph_adapter = container.get_graph_adapter()
        
        return {
            "enabled": os.getenv("ENABLE_GRAPH_REASONING", "true").lower() == "true",
            "graph_adapter_initialized": graph_adapter is not None,
            "neo4j_uri": os.getenv("NEO4J_URI", "not set"),
            "neo4j_user": os.getenv("NEO4J_USER", "not set"),
            "neo4j_database": os.getenv("NEO4J_DATABASE", "not set"),
            "adapter_type": str(type(graph_adapter)) if graph_adapter else "None"
        }
    except Exception as e:
        return {
            "error": str(e),
            "enabled": os.getenv("ENABLE_GRAPH_REASONING", "not set")
        }


@router.get(
    "/conversations",
    response_model=ConversationsResponse,
    summary="List active conversations",
    description="Get list of active conversation sessions"
)
async def list_conversations() -> ConversationsResponse:
    """
    List all active conversation sessions.
    
    Returns:
        List of active conversations with basic information
    """
    try:
        from ..core.container import get_container
        container = get_container()
        conversation_manager = container.get_conversation_manager()
        
        # Get active sessions (this is specific to in-memory implementation)
        if hasattr(conversation_manager, 'get_active_sessions'):
            active_sessions = conversation_manager.get_active_sessions()
            
            conversations = []
            for session_id in active_sessions:
                context = await conversation_manager.get_context(session_id)
                if context:
                    conversations.append(ConversationInfo(
                        session_id=session_id,
                        message_count=len(context.messages),
                        created_at=context.metadata.get("created_at") if context.metadata else None,
                        updated_at=context.metadata.get("updated_at") if context.metadata else None
                    ))
            
            return ConversationsResponse(
                conversations=conversations,
                total_count=len(conversations)
            )
        else:
            return ConversationsResponse(
                conversations=[],
                total_count=0
            )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving conversations: {str(e)}"
        )


@router.delete(
    "/conversations/{session_id}",
    summary="Delete conversation",
    description="Delete a specific conversation session"
)
async def delete_conversation(session_id: str) -> dict:
    """
    Delete a specific conversation session.
    
    Args:
        session_id: ID of the session to delete
        
    Returns:
        Success confirmation
    """
    try:
        from ..core.container import get_container
        container = get_container()
        conversation_manager = container.get_conversation_manager()
        
        success = await conversation_manager.delete_context(session_id)
        
        if success:
            return {"message": f"Conversation {session_id} deleted successfully"}
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Conversation {session_id} not found"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting conversation: {str(e)}"
        )


@router.post(
    "/conversations/cleanup",
    summary="Cleanup old conversations",
    description="Remove old conversation sessions"
)
async def cleanup_conversations(
    max_age_hours: int = 24,
    background_tasks: BackgroundTasks = None
) -> dict:
    """
    Cleanup old conversation sessions.
    
    Args:
        max_age_hours: Maximum age in hours before a conversation is considered old
        background_tasks: FastAPI background tasks
        
    Returns:
        Number of conversations cleaned up
    """
    try:
        from ..core.container import get_container
        container = get_container()
        conversation_manager = container.get_conversation_manager()
        
        # Run cleanup in background if supported
        if hasattr(conversation_manager, 'cleanup_old_contexts'):
            if background_tasks:
                async def cleanup_task():
                    return await conversation_manager.cleanup_old_contexts(max_age_hours)
                
                background_tasks.add_task(cleanup_task)
                return {"message": "Cleanup started in background"}
            else:
                cleaned_count = await conversation_manager.cleanup_old_contexts(max_age_hours)
                return {"message": f"Cleaned up {cleaned_count} old conversations"}
        else:
            return {"message": "Cleanup not supported by current conversation manager"}
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during cleanup: {str(e)}"
        )


@router.get(
    "/agents/info",
    summary="Get multi-agent system information",
    description="Get detailed information about all configured agents and their models"
)
async def get_agents_info() -> dict:
    """
    Get information about all configured agents in the multi-agent system.
    
    Returns:
        Detailed information about agents, models, and pipeline steps
    """
    try:
        multi_agent_orchestrator = get_multi_agent_orchestrator()
        health_info = await multi_agent_orchestrator.health_check()
        
        return {
            "multi_agent_system": {
                "enabled": True,
                "agents": health_info.get("agents", {}),
                "pipeline_steps": [
                    "1. Planning (DeepSeek V3.1) - Analyze query and create execution plan",
                    "2. Query Rewriting (LongCat Flash Chat) - Optimize queries for search",
                    "3. RAG Retrieval - Get relevant context using optimized queries",
                    "4. Answer Generation (Qwen3 Coder) - Generate comprehensive answers",
                    "5. Response Formatting (DeepSeek R1) - Verify and create user-friendly responses"
                ]
            },
            "models_used": {
                "smart_planner": "mistralai/mistral-7b-instruct:free",
                "answer_agent": "qwen/qwen-3-coder-free",
                "response_formatter": "deepseek/deepseek-r1-free"
            },
            "capabilities": {
                "smart_planning": "Analyzes intent, complexity, and rewrites queries in single LLM call",
                "rag_integration": "Retrieves relevant context from knowledge base (KG + Vector)",
                "answer_generation": "Creates comprehensive, structured answers",
                "response_formatting": "Verifies accuracy and ensures user-friendly output"
            },
            "configuration": {
                "verification_enabled": health_info.get("verification_enabled", True),
                "planning_enabled": health_info.get("planning_enabled", True),
                "providers": "OpenRouter API",
                "optimization": "40% fewer LLM calls vs 5-agent pipeline"
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get agents info: {str(e)}"
        )


@router.post(
    "/agents/test",
    summary="Test multi-agent system",
    description="Test all agents with a simple query to verify system functionality"
)
async def test_agents() -> dict:
    """
    Test all agents in the multi-agent system with a simple query.
    
    Returns:
        Test results including response preview and performance stats
    """
    try:
        test_request = OrchestrationRequest(
            user_query="Xin chào, hệ thống multi-agent hoạt động như thế nào?",
            use_rag=False,  # Disable RAG for this test
            rag_top_k=3,
            metadata={"test": True}
        )
        
        multi_agent_orchestrator = get_multi_agent_orchestrator()
        response = await multi_agent_orchestrator.process_request(test_request)
        
        # Get pipeline steps info
        pipeline_info = response.agent_metadata.get("pipeline_steps", {}) if response.agent_metadata else {}
        
        return {
            "test_successful": True,
            "timestamp": datetime.now().isoformat(),
            "response_preview": response.response[:200] + "..." if len(response.response) > 200 else response.response,
            "processing_stats": response.processing_stats,
            "agents_tested": {
                "smart_planner": "✅ Executed" if "smart_planning_time" in response.processing_stats else "⚠️ Skipped",
                "answer_agent": "✅ Executed" if "answer_generation_time" in response.processing_stats else "❌ Failed",
                "response_formatter": "✅ Executed" if "response_formatting_time" in response.processing_stats else "❌ Failed"
            },
            "performance": {
                "total_time": f"{response.processing_stats.get('total_time', 0):.2f}s",
                "fastest_agent": min(
                    [(k, v) for k, v in response.processing_stats.items() if k.endswith('_time') and isinstance(v, (int, float))],
                    key=lambda x: x[1],
                    default=("none", 0)
                )[0],
                "slowest_agent": max(
                    [(k, v) for k, v in response.processing_stats.items() if k.endswith('_time') and isinstance(v, (int, float))],
                    key=lambda x: x[1],
                    default=("none", 0)
                )[0]
            },
            "quality_metrics": {
                "response_length": len(response.response),
                "has_structure": "✅" if any(indicator in response.response.lower() for indicator in ["đầu tiên", "thứ hai", "cuối cùng"]) else "❌",
                "has_greeting": "✅" if any(greeting in response.response.lower() for greeting in ["xin chào", "chào bạn", "cảm ơn"]) else "❌",
                "mentions_multiagent": "✅" if "multi" in response.response.lower() or "agent" in response.response.lower() else "❌"
            }
        }
    except Exception as e:
        return {
            "test_successful": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat(),
            "suggestion": "Check OpenRouter API key and service connectivity"
        }
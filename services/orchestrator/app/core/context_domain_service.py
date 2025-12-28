"""
Pure domain service for context processing.

This service handles context processing without presentation concerns.
"""

import logging
from typing import List, Dict, Any, Optional
from ..core.domain import RAGContext

logger = logging.getLogger(__name__)

# System prompt for contextual query rewriting
CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just rewrite it if needed, otherwise return it as is."""


class ContextDomainService:
    """Pure domain service for context processing."""
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the context domain service.
        
        Args:
            llm_client: Optional LLM client for query rewriting
        """
        self._llm_client = llm_client
    
    def set_llm_client(self, llm_client: Any) -> None:
        """Set the LLM client for query rewriting."""
        self._llm_client = llm_client
    
    async def contextualize_query(
        self, 
        current_query: str, 
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Rewrite a follow-up question to be standalone using conversation context.
        
        This method handles follow-up questions that reference previous context
        (e.g., "What about that?" -> "What are the details about the UIT scholarship?")
        
        Args:
            current_query: The current user query (may contain pronouns/references)
            chat_history: List of previous messages [{"role": "user/assistant", "content": "..."}]
            
        Returns:
            Rewritten standalone query, or original query if no rewriting needed
        """
        # If no history, return original query
        if not chat_history:
            logger.debug("No chat history, returning original query")
            return current_query
        
        # If no LLM client configured, return original query
        if self._llm_client is None:
            logger.warning("No LLM client configured for query rewriting, returning original query")
            return current_query
        
        try:
            # Format chat history for the prompt
            history_text = self._format_history_for_prompt(chat_history)
            
            # Construct the user prompt
            user_prompt = f"""Chat History:
{history_text}

Latest Question: {current_query}

Standalone Question:"""
            
            # Call LLM to rewrite the query
            messages = [
                {"role": "system", "content": CONTEXTUALIZE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
            
            # Use streaming=False for quick response
            rewritten_query = await self._call_llm(messages)
            
            # Clean up the response
            rewritten_query = rewritten_query.strip()
            
            # If the response is empty or too different, return original
            if not rewritten_query or len(rewritten_query) > len(current_query) * 5:
                logger.warning("Query rewriting produced invalid result, using original")
                return current_query
            
            logger.info(f"Query rewritten: '{current_query}' -> '{rewritten_query}'")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            return current_query
    
    def _format_history_for_prompt(self, chat_history: List[Dict[str, str]]) -> str:
        """Format chat history as a readable string for the prompt."""
        lines = []
        for msg in chat_history[-6:]:  # Only use last 6 messages for context
            role = msg.get("role", "user").capitalize()
            content = msg.get("content", "")
            # Truncate very long messages
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    async def _call_llm(self, messages: List[Dict[str, str]]) -> str:
        """
        Call the LLM to get a response.
        
        Args:
            messages: List of messages for the LLM
            
        Returns:
            LLM response text
        """
        # Try different LLM client interfaces
        if hasattr(self._llm_client, 'generate'):
            # OpenRouter adapter - returns object with .content attribute
            response = await self._llm_client.generate(
                messages=messages,
                max_tokens=150,
                temperature=0.0  # Deterministic for query rewriting
            )
            # Handle different response types
            if hasattr(response, 'content'):
                return response.content
            elif isinstance(response, dict):
                return response.get("content", "")
            else:
                return str(response)
        
        elif hasattr(self._llm_client, 'generate_response'):
            # AgentPort interface
            from ..agents.base import AgentRequest
            request = AgentRequest(
                prompt=messages[-1]["content"] if messages else "",
                context=None,
                temperature=0.0,
                max_tokens=150,
                stream=False
            )
            response = await self._llm_client.generate_response(request)
            return response.content if hasattr(response, 'content') else str(response)
        
        elif hasattr(self._llm_client, 'chat'):
            # OpenAI-style client
            response = await self._llm_client.chat(
                messages=messages,
                max_tokens=150,
                temperature=0.0
            )
            return response.choices[0].message.content if hasattr(response, 'choices') else str(response)
        
        else:
            raise ValueError("LLM client does not have a supported interface (generate/generate_response/chat)")
    
    def extract_relevant_documents(self, rag_context: RAGContext, max_docs: int = 5) -> List[Dict[str, Any]]:
        """
        Extract most relevant documents from RAG context.
        
        Args:
            rag_context: RAG context containing retrieved documents
            max_docs: Maximum number of documents to extract
            
        Returns:
            List of relevant documents with metadata
        """
        relevant_docs = []
        
        for i, doc in enumerate(rag_context.retrieved_documents[:max_docs]):
            # RAG service returns 'text' field, fallback to 'content'
            content = doc.get("text", doc.get("content", "")).strip()
            
            if content and len(content) > 10:  # Filter out very short content
                relevant_docs.append({
                    "rank": i + 1,
                    "content": content,
                    "title": doc.get("title", f"Document {i + 1}"),
                    "metadata": doc.get("metadata", {}),
                    "relevance_score": rag_context.relevance_scores[i] if rag_context.relevance_scores and i < len(rag_context.relevance_scores) else 0.0
                })
        
        return relevant_docs
    
    def assess_context_quality(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Assess the quality of retrieved context.
        
        Args:
            documents: List of relevant documents
            
        Returns:
            Quality assessment metrics
        """
        if not documents:
            return {
                "quality_score": 0.0,
                "has_sufficient_content": False,
                "average_relevance": 0.0,
                "document_count": 0
            }
        
        # RAG service returns 'text' field, fallback to 'content'
        total_content_length = sum(len(doc.get("text", doc.get("content", ""))) for doc in documents)
        average_relevance = sum(doc.get("relevance_score", 0.0) for doc in documents) / len(documents)
        
        quality_score = min(
            (average_relevance * 0.6) + 
            (min(total_content_length / 1000, 1.0) * 0.4), 
            1.0
        )
        
        return {
            "quality_score": quality_score,
            "has_sufficient_content": total_content_length > 100,
            "average_relevance": average_relevance,
            "document_count": len(documents),
            "total_content_length": total_content_length
        }
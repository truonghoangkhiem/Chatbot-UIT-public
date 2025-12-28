"""
Answer Agent implementation.

This agent generates comprehensive answers based on retrieved context and user queries.
Uses Qwen3 Coder model for structured reasoning and answer generation.

Enhanced with:
- Detailed source citations with char_spans
- Document metadata (doc_type, faculty, year, subject)
- Streaming support for real-time response generation
"""

import json
from typing import Dict, Any, List, AsyncGenerator
from ..agents.base import SpecializedAgent, AgentConfig, AgentType, AnswerResult, DetailedSource


class AnswerAgent(SpecializedAgent):
    """
    Answer Agent responsible for generating comprehensive answers from context.
    
    This agent:
    1. Analyzes retrieved documents and context
    2. Synthesizes information from multiple sources
    3. Generates structured, comprehensive answers
    4. Provides reasoning steps and source citations
    5. Handles Vietnamese academic language appropriately
    """
    
    def __init__(self, config: AgentConfig, agent_port):
        """
        Initialize the Answer Agent.
        
        Args:
            config: Agent configuration containing model settings and parameters
            agent_port: Port for communicating with the underlying LLM
        """
        super().__init__(config, agent_port)
        
        # Extract agent-specific parameters from config
        params = getattr(config, 'parameters', {})
        self.min_answer_length = params.get('min_answer_length', 50)
        self.max_sources = params.get('max_sources', 5)
        self.confidence_thresholds = params.get('confidence_thresholds', {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        })
    
    async def process(self, input_data: Dict[str, Any]) -> AnswerResult:
        """
        Generate comprehensive answer from context and query.
        
        Args:
            input_data: Dictionary containing:
                - query: str - User query
                - context_documents: List[Dict] - Retrieved documents
                - rewritten_queries: Optional[List[str]] - Alternative queries
                - previous_context: Optional[str] - Conversation context
                - previous_feedback: Optional[str] - Feedback from ResponseFormatter for retry
        
        Returns:
            AnswerResult containing the generated answer and metadata
        """
        query = input_data.get("query", "")
        context_documents = input_data.get("context_documents", [])
        rewritten_queries = input_data.get("rewritten_queries", [])
        previous_context = input_data.get("previous_context", "")
        previous_feedback = input_data.get("previous_feedback", "")  # NEW: Feedback for retry
        
        # Build the answer generation prompt
        prompt = self._build_answer_prompt(
            query, context_documents, rewritten_queries, previous_context, previous_feedback
        )
        
        # Get response from the agent
        response = await self._make_agent_request(prompt)
        
        # Parse the response
        try:
            answer_data = json.loads(response.content)
            return self._create_answer_result(answer_data, query)
        except json.JSONDecodeError:
            # Fallback to extracting answer from text response
            return self._create_fallback_answer(query, response.content, context_documents)
    
    async def stream_process(self, input_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """
        Stream comprehensive answer generation from context and query.
        
        This method yields answer text chunks in real-time as they are generated,
        providing better user experience for long-running queries.
        
        Args:
            input_data: Dictionary containing:
                - query: str - User query
                - context_documents: List[Dict] - Retrieved documents
                - rewritten_queries: Optional[List[str]] - Alternative queries
                - previous_context: Optional[str] - Conversation context
                - previous_feedback: Optional[str] - Feedback from ResponseFormatter for retry
        
        Yields:
            String chunks of the generated answer as they become available
        """
        query = input_data.get("query", "")
        context_documents = input_data.get("context_documents", [])
        rewritten_queries = input_data.get("rewritten_queries", [])
        previous_context = input_data.get("previous_context", "")
        previous_feedback = input_data.get("previous_feedback", "")
        
        # Build the answer generation prompt
        prompt = self._build_answer_prompt(
            query, context_documents, rewritten_queries, previous_context, previous_feedback
        )
        
        # Stream response from the agent
        async for chunk in self._stream_agent_request(prompt):
            yield chunk
    
    async def _stream_agent_request(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Stream a request to the underlying agent.
        
        Args:
            prompt: The prompt to send to the agent
            
        Yields:
            String chunks from the agent response
        """
        from ..core.domain import ConversationContext, AgentRequest
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        
        # Log input if debug mode is enabled
        if os.getenv('LOG_LEVEL', 'INFO').upper() == 'DEBUG':
            logger.debug(f"\n{'='*80}")
            logger.debug(f"🔵 AGENT STREAMING INPUT - {self.config.agent_type.value.upper()}")
            logger.debug(f"{'='*80}")
            logger.debug(f"System Prompt Length: {len(self.config.system_prompt)} chars")
            logger.debug(f"User Prompt: {prompt[:500]}..." if len(prompt) > 500 else f"User Prompt: {prompt}")
            logger.debug(f"Model: {self.config.model}")
            logger.debug(f"Temperature: {self.config.temperature}")
            logger.debug(f"{'='*80}\n")
        
        # Create conversation context with system prompt from agent config
        conversation_context = ConversationContext(
            session_id="agent_stream_session",
            messages=[],
            system_prompt=self.config.system_prompt,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        request = AgentRequest(
            prompt=prompt,
            context=conversation_context,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            stream=True,  # Enable streaming
            metadata={"agent_type": self.config.agent_type.value}
        )
        
        # Stream response from agent port
        async for chunk in self.agent_port.stream_response(request):
            yield chunk
    
    def _filter_amended_documents(
        self,
        context_documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter documents to prioritize amended content and remove duplicates.
        
        When the same article appears multiple times (old and new versions),
        this method keeps only the newest version (marked with is_amended=True
        or from amendment documents).
        
        Args:
            context_documents: List of retrieved documents
            
        Returns:
            Filtered list with duplicates removed, prioritizing amended content
        """
        import re
        
        # Guard: return empty if no documents
        if not context_documents:
            return []
        
        # Track which articles we've seen and their best version
        article_map = {}  # article_number -> (doc, is_amended)
        other_docs = []  # Non-article documents
        
        for doc in context_documents:
            # Skip None documents
            if doc is None:
                continue
                
            title = doc.get("title", "") or ""
            content = doc.get("content", "") or doc.get("text", "") or ""
            is_amended = doc.get("is_amended", False)
            
            # Check if this is an article (Điều X)
            article_match = re.search(r'Điều\s*(\d+)', title, re.IGNORECASE)
            
            if article_match:
                article_num = article_match.group(1)
                
                # Skip if content contains OLD markers that should be superseded
                old_markers = [
                    "ĐTBC ≥ 7" in content or "ĐTBC >= 7" in content,
                    "điểm trung bình chung ≥ 7" in content.lower(),
                    "học vượt chỉ dành cho sinh viên có ĐTBC" in content,
                ]
                has_old_markers = any(old_markers)
                
                # Check if this is amendment content
                new_markers = [
                    "is_amended" in doc and doc["is_amended"],
                    "Mục" in title and "Điều" in title,  # e.g., "Mục b khoản 1 Điều 14"
                    doc.get("source", "") == "knowledge_graph",
                ]
                is_new_content = any(new_markers)
                
                if article_num in article_map:
                    existing_doc, existing_is_amended = article_map[article_num]
                    # Replace if current is amended and existing is not
                    if is_new_content and not existing_is_amended:
                        article_map[article_num] = (doc, is_new_content)
                    # Replace if existing has old markers and current doesn't
                    elif has_old_markers:
                        pass  # Keep existing
                    elif not is_new_content and existing_is_amended:
                        pass  # Keep existing amended version
                else:
                    # First time seeing this article
                    if not has_old_markers or is_new_content:
                        article_map[article_num] = (doc, is_new_content)
            else:
                # Not an article document
                other_docs.append(doc)
        
        # Combine filtered articles with other docs
        filtered = [doc for doc, _ in article_map.values()] + other_docs
        return filtered
    
    def _build_answer_prompt(
        self,
        query: str,
        context_documents: List[Dict[str, Any]],
        rewritten_queries: List[str],
        previous_context: str,
        previous_feedback: str = ""
    ) -> str:
        """
        Build the comprehensive answer generation prompt.
        
        Args:
            query: User's original query
            context_documents: Retrieved documents with content
            rewritten_queries: Alternative query formulations
            previous_context: Conversation context
            previous_feedback: Feedback from ResponseFormatter for improving answer (retry scenario)
        """
        prompt_parts = []
        
        # IMPORTANT: Add feedback section at the top if this is a retry
        if previous_feedback:
            prompt_parts.append("=" * 60)
            prompt_parts.append("⚠️ IMPROVEMENT REQUEST - PLEASE REVISE YOUR PREVIOUS ANSWER")
            prompt_parts.append("=" * 60)
            prompt_parts.append("")
            prompt_parts.append("Your previous answer was evaluated and needs improvement.")
            prompt_parts.append("Please carefully read the feedback below and generate a better answer:")
            prompt_parts.append("")
            prompt_parts.append(previous_feedback)
            prompt_parts.append("")
            prompt_parts.append("=" * 60)
            prompt_parts.append("Now, please provide an improved answer addressing the feedback above.")
            prompt_parts.append("=" * 60)
            prompt_parts.append("")
        
        prompt_parts.append(f"Query: {query}")
        
        if rewritten_queries:
            prompt_parts.append(f"Query Variations: {', '.join(rewritten_queries)}")
        
        if previous_context:
            prompt_parts.append(f"Context: {previous_context}")
        
        # Filter documents to remove old versions when amendments exist
        filtered_documents = self._filter_amended_documents(context_documents)
        
        # Add context documents with full content
        if filtered_documents:
            prompt_parts.append("\nDocuments:")
            for i, doc in enumerate(filtered_documents, 1):
                title = doc.get("title", f"Document {i}")
                content = doc.get("content", "") or doc.get("text", "")
                score = doc.get("score", 0.0)
                is_amended = doc.get("is_amended", False)
                
                # Mark amended documents for clarity
                amended_marker = " [UPDATED VERSION]" if is_amended else ""
                prompt_parts.append(f"[{i}] {title}{amended_marker} (Score: {score:.2f})")
                prompt_parts.append(content)
        
        return "\n".join(prompt_parts)
    
    def _create_answer_result(self, answer_data: Dict[str, Any], original_query: str) -> AnswerResult:
        """Create AnswerResult from parsed JSON data."""
        return AnswerResult(
            query=original_query,
            answer=answer_data.get("answer", ""),
            confidence=answer_data.get("confidence", 0.5),
            sources_used=answer_data.get("sources_used", []),
            reasoning_steps=answer_data.get("reasoning_steps", []),
            metadata=answer_data.get("metadata", {}),
            detailed_sources=[]  # JSON response doesn't include detailed sources
        )
    
    def _create_fallback_answer(
        self, 
        query: str, 
        response_content: str, 
        context_documents: List[Dict[str, Any]]
    ) -> AnswerResult:
        """Create fallback answer when JSON parsing fails."""
        # Extract the main answer from response content
        answer = self._extract_answer_from_text(response_content)
        
        # Determine confidence based on context quality
        confidence = self._estimate_confidence(context_documents, answer)
        
        # Extract sources from context documents using configured max_sources
        sources_used = [
            doc.get("title", f"Document {i+1}")
            for i, doc in enumerate(context_documents[:self.max_sources])
        ]
        
        # Create detailed sources with citation information
        detailed_sources = self._create_detailed_sources(context_documents[:self.max_sources])
        
        return AnswerResult(
            query=query,
            answer=answer,
            confidence=confidence,
            sources_used=sources_used,
            reasoning_steps=["Sử dụng thông tin từ tài liệu tham khảo", "Tổng hợp và phân tích nội dung"],
            metadata={
                "fallback": True,
                "method": "text_extraction",
                "original_length": len(response_content)
            },
            detailed_sources=detailed_sources
        )
    
    def _create_detailed_sources(self, context_documents: List[Dict[str, Any]]) -> List[DetailedSource]:
        """
        Create detailed source citations from context documents.
        
        Args:
            context_documents: List of retrieved documents with citation data
            
        Returns:
            List of DetailedSource objects with citation information
        """
        detailed_sources = []
        
        for doc in context_documents:
            # Get basic info
            title = doc.get("title", "Unknown")
            score = doc.get("score", 0.0)
            
            # Get document ID and chunk info
            meta = doc.get("meta", doc.get("metadata", {}))
            doc_id = meta.get("doc_id")
            chunk_id = meta.get("chunk_id")
            
            # Get citation data
            citation = doc.get("citation", {})
            char_spans = doc.get("char_spans", [])
            highlighted_text = doc.get("highlighted_text", [])
            
            # If citation is a dict, extract char_spans from it
            if isinstance(citation, dict):
                if not char_spans and citation.get("char_spans"):
                    char_spans = citation.get("char_spans", [])
                if not highlighted_text and citation.get("highlighted_text"):
                    highlighted_text = citation.get("highlighted_text", [])
            
            # Get document classification
            doc_type = doc.get("doc_type")
            faculty = doc.get("faculty")
            year = doc.get("year")
            subject = doc.get("subject")
            
            # Create citation text from highlighted spans or first char_span
            citation_text = None
            if highlighted_text:
                citation_text = highlighted_text[0] if isinstance(highlighted_text, list) else highlighted_text
            elif char_spans and isinstance(char_spans, list) and len(char_spans) > 0:
                first_span = char_spans[0]
                if isinstance(first_span, dict):
                    citation_text = first_span.get("text", "")
            
            detailed_source = DetailedSource(
                title=title,
                doc_id=doc_id,
                chunk_id=chunk_id,
                score=score,
                citation_text=citation_text,
                char_spans=char_spans if char_spans else None,
                highlighted_text=highlighted_text if highlighted_text else None,
                doc_type=doc_type,
                faculty=faculty,
                year=year,
                subject=subject
            )
            
            detailed_sources.append(detailed_source)
        
        return detailed_sources
    
    def _extract_answer_from_text(self, text: str) -> str:
        """Extract the main answer from text response."""
        # Try to find and parse JSON first
        import re
        
        # Look for JSON blocks (with or without ```json markers)
        json_pattern = r'```json\s*({.*?})\s*```|({.*?})'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        if matches:
            # Get the first non-empty match
            json_str = next((m for group in matches for m in group if m), None)
            if json_str:
                try:
                    data = json.loads(json_str)
                    # Extract answer field from JSON
                    if isinstance(data, dict) and "answer" in data:
                        return data["answer"]
                except json.JSONDecodeError:
                    pass
        
        # --- BẮT ĐẦU PHẦN SỬA ---
        # Fallback: Clean up markers only, PRESERVE NEWLINES and SPACING
        clean_text = text.strip()
        
        # Remove opening ```json or ``` markers
        clean_text = re.sub(r'^```[a-zA-Z]*\s*\n', '', clean_text)
        
        # Remove closing ``` markers
        clean_text = re.sub(r'\n\s*```\s*$', '', clean_text)
        
        # Remove standalone JSON braces if they wrap the entire text (rare edge case)
        if clean_text.startswith('{') and clean_text.endswith('}'):
             # Chỉ remove nếu nó trông giống JSON wrapper bị lỗi, 
             # nhưng nếu là text chat bình thường bắt đầu bằng { thì giữ nguyên
             pass 

        answer = clean_text
        # --- KẾT THÚC PHẦN SỬA ---
        
        # Ensure minimum answer length using configured threshold
        if len(answer) < self.min_answer_length:
            answer = f"Dựa trên thông tin có sẵn: {answer}"
        
        return answer
    
    def _estimate_confidence(self, context_documents: List[Dict[str, Any]], answer: str) -> float:
        """Estimate confidence based on context quality and answer characteristics."""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on number of relevant documents
        if context_documents:
            doc_count = len(context_documents)
            confidence += min(0.2, doc_count * 0.05)
            
            # Check document relevance scores
            avg_score = sum(doc.get("score", 0) for doc in context_documents) / doc_count
            confidence += avg_score * 0.3
        
        # Boost confidence based on answer length and structure
        if len(answer) > 100:
            confidence += 0.1
        
        if any(keyword in answer.lower() for keyword in ["uit", "trường", "đại học", "quy định"]):
            confidence += 0.1
        
        # Cap confidence at reasonable levels
        return min(0.95, max(0.1, confidence))
    
    def _analyze_answer_type(self, query: str, answer: str) -> str:
        """Analyze the type of answer generated."""
        query_lower = query.lower()
        answer_lower = answer.lower()
        
        # Procedural answers (how-to questions)
        if any(word in query_lower for word in ["làm thế nào", "hướng dẫn", "cách", "thủ tục", "quy trình"]):
            return "procedural"
        
        # Comparative answers
        if any(word in query_lower for word in ["so sánh", "khác nhau", "giống", "phân biệt"]):
            return "comparative"
        
        # Informative answers (default)
        return "informative"
    
    def _assess_completeness(self, answer: str, context_documents: List[Dict[str, Any]]) -> str:
        """Assess the completeness of the generated answer."""
        if not context_documents:
            return "insufficient_data"
        
        # Basic heuristics for completeness using configured thresholds
        min_partial_length = self.min_answer_length * 2  # 2x minimum for partial
        min_complete_length = self.min_answer_length * 6  # 6x minimum for complete
        
        if len(answer) < min_partial_length:
            return "partial"
        
        # Check if answer addresses multiple aspects
        aspect_indicators = ["đầu tiên", "thứ hai", "cuối cùng", "ngoài ra", "bên cạnh đó"]
        if any(indicator in answer.lower() for indicator in aspect_indicators):
            return "complete"
        
        return "partial" if len(answer) < min_complete_length else "complete"
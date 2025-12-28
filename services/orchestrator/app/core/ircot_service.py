"""
IRCoT (Interleaving Retrieval with Chain-of-Thought) Reasoning Service.

This service implements the IRCoT algorithm for complex multi-hop questions,
enabling dynamic retrieval based on chain-of-thought reasoning steps.

Reference: "Interleaving Retrieval with Chain-of-Thought Reasoning for 
            Knowledge-Intensive Multi-Step Questions" (Trivedi et al., 2022)

Key Innovation:
- Instead of retrieving once and then reasoning (standard RAG),
  IRCoT interleaves retrieval with reasoning steps:
  
  Standard RAG: Retrieve → Reason → Answer
  IRCoT: Retrieve → Reason → Retrieve → Reason → ... → Answer
"""

import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from .ircot_config import (
    IRCoTConfig, 
    IRCoTIterationResult, 
    IRCoTResult,
    DEFAULT_IRCOT_CONFIG
)
from ..ports.agent_ports import AgentPort, RAGServicePort
from ..core.domain import AgentRequest, ConversationContext

logger = logging.getLogger(__name__)


class IRCoTReasoningService:
    """
    Service implementing IRCoT (Interleaving Retrieval with Chain-of-Thought).
    
    This service enables iterative reasoning with dynamic retrieval for
    complex, multi-hop questions that cannot be answered with single retrieval.
    
    Algorithm:
    1. Initial retrieval based on original query
    2. Generate first CoT reasoning step
    3. If more information needed, generate new search query from reasoning
    4. Retrieve additional context with new query
    5. Repeat until confident or max iterations reached
    6. Return accumulated context for final answer generation
    """
    
    def __init__(
        self,
        agent_port: AgentPort,
        rag_port: RAGServicePort,
        config: Optional[IRCoTConfig] = None
    ):
        """
        Initialize the IRCoT Reasoning Service.
        
        Args:
            agent_port: Port for LLM communication (for CoT generation)
            rag_port: Port for RAG retrieval
            config: IRCoT configuration (uses defaults if not provided)
        """
        self.agent_port = agent_port
        self.rag_port = rag_port
        self.config = config or DEFAULT_IRCOT_CONFIG
        
        # System prompt for CoT reasoning
        self.cot_system_prompt = """Bạn là một trợ lý AI chuyên về suy luận logic và phân tích thông tin.

Nhiệm vụ của bạn là:
1. Phân tích thông tin được cung cấp
2. Thực hiện một bước suy luận logic
3. Xác định những thông tin còn thiếu
4. Đề xuất câu truy vấn để tìm thêm thông tin (nếu cần)

Luôn trả lời bằng JSON hợp lệ theo format yêu cầu.
Sử dụng tiếng Việt cho nội dung reasoning và thông tin."""
    
    async def reason_with_retrieval(
        self,
        query: str,
        initial_context: Optional[List[Dict[str, Any]]] = None,
        extracted_filters: Optional[Any] = None,
        max_iterations_override: Optional[int] = None
    ) -> IRCoTResult:
        """
        Execute IRCoT reasoning with dynamic retrieval.
        
        Args:
            query: The original user query
            initial_context: Optional initial context documents
            extracted_filters: Optional filters for RAG retrieval
            max_iterations_override: Override max iterations (for complexity-based optimization)
            
        Returns:
            IRCoTResult containing all iterations and accumulated context
        """
        start_time = time.time()
        
        # Use override if provided, otherwise use config default
        effective_max_iterations = max_iterations_override or self.config.max_iterations
        
        logger.info(f"🔄 Starting IRCoT reasoning for: {query[:50]}... (max_iter={effective_max_iterations})")
        
        # Initialize accumulated context
        accumulated_context: List[Dict[str, Any]] = []
        if initial_context:
            accumulated_context.extend(initial_context)
            logger.info(f"  Initial context: {len(initial_context)} documents")
        
        iterations: List[IRCoTIterationResult] = []
        reasoning_steps: List[str] = []
        final_confidence = 0.0
        early_stopped = False
        
        # Current search query (starts with original)
        current_search_query = query
        
        for iteration_num in range(1, effective_max_iterations + 1):
            logger.info(f"  📍 Iteration {iteration_num}/{effective_max_iterations}")
            
            # Step 1: Retrieve with current query (skip if first iteration with initial context)
            if iteration_num > 1 or not initial_context:
                new_docs = await self._retrieve_documents(
                    current_search_query,
                    extracted_filters
                )
                
                # Add non-duplicate documents
                new_count = self._add_unique_documents(accumulated_context, new_docs)
                logger.info(f"    Retrieved {len(new_docs)} docs, {new_count} new unique")
            
            # Check context limit
            if len(accumulated_context) >= self.config.max_context_documents:
                accumulated_context = accumulated_context[:self.config.max_context_documents]
                logger.info(f"    Context limit reached: {self.config.max_context_documents}")
            
            # Step 2: Generate CoT reasoning step
            cot_result = await self._generate_cot_step(
                original_query=query,
                accumulated_context=accumulated_context,
                previous_reasoning=reasoning_steps,
                current_step=iteration_num,
                max_steps=effective_max_iterations
            )
            
            # Record iteration
            iteration_result = IRCoTIterationResult(
                iteration=iteration_num,
                reasoning_step=cot_result.get("reasoning_step", ""),
                next_search_query=cot_result.get("next_search_query"),
                retrieved_documents=[],  # Store doc refs separately
                confidence=cot_result.get("confidence", 0.0),
                can_answer_now=cot_result.get("can_answer_now", False),
                information_gaps=cot_result.get("information_gaps", [])
            )
            iterations.append(iteration_result)
            
            if iteration_result.reasoning_step:
                reasoning_steps.append(iteration_result.reasoning_step)
            
            final_confidence = iteration_result.confidence
            
            logger.info(f"    Confidence: {final_confidence:.2f}")
            logger.info(f"    Can answer now: {iteration_result.can_answer_now}")
            
            # Step 3: Check early stopping conditions
            if self._should_stop_early(iteration_result, iteration_num):
                early_stopped = True
                logger.info(f"  ⏹️ Early stopping at iteration {iteration_num}")
                break
            
            # Step 4: Update search query for next iteration
            if iteration_result.next_search_query:
                current_search_query = iteration_result.next_search_query
                logger.info(f"    Next query: {current_search_query[:50]}...")
            else:
                # No more queries needed
                logger.info(f"    No more retrieval needed")
                break
        
        processing_time = time.time() - start_time
        
        # Compile final reasoning
        final_reasoning = self._compile_final_reasoning(reasoning_steps)
        
        result = IRCoTResult(
            original_query=query,
            iterations=iterations,
            total_iterations=len(iterations),
            accumulated_context=accumulated_context,
            final_reasoning=final_reasoning,
            final_confidence=final_confidence,
            early_stopped=early_stopped,
            processing_time=processing_time
        )
        
        logger.info(f"✅ IRCoT completed: {len(iterations)} iterations, "
                   f"{len(accumulated_context)} docs, confidence={final_confidence:.2f}")
        
        return result
    
    async def _retrieve_documents(
        self,
        query: str,
        filters: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve documents for a query."""
        try:
            result = await self.rag_port.retrieve_context(
                query=query,
                top_k=self.config.retrieval_top_k,
                filters=filters,
                use_rerank=True
            )
            return result.get("retrieved_documents", [])
        except Exception as e:
            logger.error(f"IRCoT retrieval error: {e}")
            return []
    
    def _add_unique_documents(
        self,
        accumulated: List[Dict[str, Any]],
        new_docs: List[Dict[str, Any]]
    ) -> int:
        """Add unique documents to accumulated list. Returns count of added docs."""
        if not self.config.deduplication_enabled:
            accumulated.extend(new_docs)
            return len(new_docs)
        
        # Get existing content signatures
        existing_signatures = set()
        for doc in accumulated:
            content = doc.get("text", doc.get("content", ""))
            sig = content[:100].strip().lower()
            existing_signatures.add(sig)
        
        added_count = 0
        for doc in new_docs:
            content = doc.get("text", doc.get("content", ""))
            sig = content[:100].strip().lower()
            
            if sig and sig not in existing_signatures:
                accumulated.append(doc)
                existing_signatures.add(sig)
                added_count += 1
        
        return added_count
    
    async def _generate_cot_step(
        self,
        original_query: str,
        accumulated_context: List[Dict[str, Any]],
        previous_reasoning: List[str],
        current_step: int,
        max_steps: int
    ) -> Dict[str, Any]:
        """Generate a single Chain-of-Thought reasoning step."""
        
        # Format accumulated context
        context_text = self._format_context_for_cot(accumulated_context)
        
        # Format previous reasoning
        prev_reasoning_text = ""
        if previous_reasoning:
            prev_reasoning_text = "\n".join([
                f"Bước {i+1}: {step}" 
                for i, step in enumerate(previous_reasoning)
            ])
        
        # Build prompt from template
        prompt = self.config.reasoning_prompt_template.format(
            original_query=original_query,
            accumulated_context=context_text,
            previous_reasoning=prev_reasoning_text,
            current_step=current_step,
            max_steps=max_steps
        )
        
        # Make LLM request
        try:
            conversation_context = ConversationContext(
                session_id="ircot_session",
                messages=[],
                system_prompt=self.cot_system_prompt,
                temperature=self.config.cot_temperature
            )
            
            request = AgentRequest(
                prompt=prompt,
                context=conversation_context,
                model=self.config.cot_model,  # Use configured model (None = default)
                temperature=self.config.cot_temperature,
                metadata={"agent_type": "ircot_reasoner"}
            )
            
            response = await self.agent_port.generate_response(request)
            
            # Parse JSON response
            return self._parse_cot_response(response.content)
            
        except Exception as e:
            logger.error(f"IRCoT CoT generation error: {e}")
            return {
                "reasoning_step": f"Lỗi khi suy luận: {str(e)}",
                "information_gaps": [],
                "next_search_query": None,
                "confidence": 0.0,
                "can_answer_now": False
            }
    
    def _format_context_for_cot(
        self, 
        documents: List[Dict[str, Any]]
    ) -> str:
        """Format documents for CoT prompt."""
        if not documents:
            return "(Chưa có thông tin)"
        
        parts = []
        for i, doc in enumerate(documents[:10], 1):  # Limit to 10 docs for prompt
            content = doc.get("text", doc.get("content", ""))[:500]  # Truncate
            title = doc.get("title", f"Tài liệu {i}")
            parts.append(f"[{i}] {title}:\n{content}")
        
        return "\n\n".join(parts)
    
    def _parse_cot_response(self, content: str) -> Dict[str, Any]:
        """Parse CoT response from LLM."""
        # Try JSON parsing
        try:
            # Find JSON in response
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
                
                return {
                    "reasoning_step": data.get("reasoning_step", ""),
                    "information_gaps": data.get("information_gaps", []),
                    "next_search_query": data.get("next_search_query"),
                    "confidence": float(data.get("confidence", 0.5)),
                    "can_answer_now": data.get("can_answer_now", False)
                }
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Fallback: extract info from text
        return self._extract_from_text(content)
    
    def _extract_from_text(self, content: str) -> Dict[str, Any]:
        """Extract CoT info from non-JSON text response."""
        # Default result
        result = {
            "reasoning_step": content[:500],  # Use first part as reasoning
            "information_gaps": [],
            "next_search_query": None,
            "confidence": 0.5,
            "can_answer_now": False
        }
        
        # Try to detect if answer is complete
        completion_markers = [
            "đủ thông tin", "có thể trả lời", "kết luận", 
            "tóm lại", "như vậy", "câu trả lời"
        ]
        content_lower = content.lower()
        
        for marker in completion_markers:
            if marker in content_lower:
                result["confidence"] = 0.75
                result["can_answer_now"] = True
                break
        
        return result
    
    def _should_stop_early(
        self, 
        iteration: IRCoTIterationResult,
        iteration_num: int
    ) -> bool:
        """Determine if reasoning should stop early.
        
        OPTIMIZED: More aggressive early stopping for faster responses.
        """
        # Must complete minimum iterations
        if iteration_num < self.config.min_iterations:
            return False
        
        # Stop if no early stopping enabled
        if not self.config.early_stopping_enabled:
            return False
        
        # OPTIMIZATION 1: Stop if can answer now with sufficient confidence
        if (iteration.can_answer_now and 
            iteration.confidence >= self.config.early_stopping_threshold):
            logger.info(f"    ⚡ Early stop: can_answer=True, confidence={iteration.confidence:.2f}")
            return True
        
        # OPTIMIZATION 2: Stop if no information gaps remain
        if iteration.can_answer_now and not iteration.information_gaps:
            logger.info(f"    ⚡ Early stop: no information gaps, can_answer=True")
            return True
        
        # OPTIMIZATION 3: Stop if no more retrieval needed
        if iteration.next_search_query is None and iteration.can_answer_now:
            logger.info(f"    ⚡ Early stop: no more queries needed")
            return True
        
        # OPTIMIZATION 4: Stop if confidence is high enough even without explicit can_answer
        if iteration.confidence >= 0.80:
            logger.info(f"    ⚡ Early stop: high confidence={iteration.confidence:.2f}")
            return True
        
        return False
    
    def _compile_final_reasoning(self, reasoning_steps: List[str]) -> str:
        """Compile all reasoning steps into final reasoning summary."""
        if not reasoning_steps:
            return ""
        
        if len(reasoning_steps) == 1:
            return reasoning_steps[0]
        
        parts = []
        for i, step in enumerate(reasoning_steps, 1):
            parts.append(f"**Bước {i}:** {step}")
        
        return "\n\n".join(parts)

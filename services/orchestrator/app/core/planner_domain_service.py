"""
Domain-driven planner service.

This service handles planning logic without depending on serialization formats.
"""

from typing import Dict, Any, List
from ..agents.base import PlanResult, PlanStep
from ..core.domain import AgentResponse


class PlannerDomainService:
    """Pure domain service for planning logic."""
    
    def parse_plan_response(self, response: AgentResponse, query: str) -> PlanResult:
        """
        Parse agent response into domain objects without JSON dependency.
        
        Args:
            response: Raw agent response
            query: Original user query
            
        Returns:
            PlanResult domain object
        """
        # Extract key information from natural language response
        content = response.content.strip()
        
        # Simple heuristic-based parsing (no JSON dependency)
        intent = self._extract_intent(content, query)
        complexity = self._assess_complexity(content, query)
        steps = self._extract_steps(content)
        
        return PlanResult(
            query=query,
            intent=intent,
            complexity=complexity,
            steps=steps,
            estimated_tokens=self._estimate_tokens(steps),
            requires_verification=complexity in ["complex", "medium"],
            metadata={
                "parsing_method": "heuristic",
                "content_length": len(content),
                "response_model": response.model_used
            }
        )
    
    def _extract_intent(self, content: str, query: str) -> str:
        """Extract user intent from content."""
        content_lower = content.lower()
        query_lower = query.lower()
        
        # Intent classification based on keywords
        if any(word in query_lower for word in ["học phí", "chi phí", "tiền"]):
            return "fee_inquiry"
        elif any(word in query_lower for word in ["đăng ký", "đky", "môn học"]):
            return "registration"
        elif any(word in query_lower for word in ["lịch", "thời gian", "khi nào"]):
            return "schedule_inquiry"
        elif any(word in query_lower for word in ["quy định", "luật", "chính sách"]):
            return "policy_inquiry"
        else:
            return "general_inquiry"
    
    def _assess_complexity(self, content: str, query: str) -> str:
        """Assess query complexity."""
        # Count question indicators
        question_count = query.count("?") + query.count("là gì") + query.count("như thế nào")
        
        # Check for multiple topics
        topic_indicators = ["và", "hoặc", "ngoài ra", "thêm", "khác"]
        has_multiple_topics = any(indicator in query.lower() for indicator in topic_indicators)
        
        # Length-based assessment
        word_count = len(query.split())
        
        if question_count > 2 or has_multiple_topics or word_count > 20:
            return "complex"
        elif question_count > 1 or word_count > 10:
            return "medium"
        else:
            return "simple"
    
    def _extract_steps(self, content: str) -> List[PlanStep]:
        """Extract execution steps from content."""
        # Default steps based on common patterns
        return [
            PlanStep(
                step_id="1",
                action="query_analysis",
                description="Phân tích và hiểu rõ câu hỏi của người dùng",
                dependencies=[],
                parameters={"priority": "high"}
            ),
            PlanStep(
                step_id="2", 
                action="information_retrieval",
                description="Tìm kiếm thông tin liên quan từ knowledge base",
                dependencies=["1"],
                parameters={"search_mode": "hybrid"}
            ),
            PlanStep(
                step_id="3",
                action="response_generation",
                description="Tạo câu trả lời dựa trên thông tin đã thu thập",
                dependencies=["2"],
                parameters={"style": "informative"}
            )
        ]
    
    def _estimate_tokens(self, steps: List[PlanStep]) -> int:
        """Estimate token usage for execution."""
        base_tokens = 100
        step_tokens = len(steps) * 50
        return base_tokens + step_tokens
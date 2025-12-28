"""
Response Formatter Agent implementation.

This agent combines the functionality of Verifier and Response agents
into a single optimized agent to reduce LLM calls.

Merged from:
- VerifierAgent: Quality assurance, fact-checking, accuracy scoring
- ResponseAgent: User-friendly formatting, tone adjustment, adding greetings
"""

import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..agents.base import SpecializedAgent, AgentConfig, AgentType


@dataclass
class FormattedResponseResult:
    """Result from the Response Formatter agent (combined verification + formatting)."""
    # Final response
    final_response: str
    
    # Verification fields
    needs_improvement: bool
    issues: List[str]
    suggestions: List[str]
    
    # Quality scores
    quality_scores: Dict[str, int]  # accuracy, completeness, friendliness (0-10)
    overall_score: float
    
    # Formatting metadata
    tone: str
    includes_greeting: bool
    includes_next_steps: bool
    
    # General metadata
    confidence: float
    metadata: Dict[str, Any]


class ResponseFormatterAgent(SpecializedAgent):
    """
    Response Formatter Agent - Optimized combination of Verifier + Response Agent.
    
    This agent performs in a single LLM call:
    1. Light verification (check for obvious errors)
    2. Quality scoring (accuracy, completeness, friendliness)
    3. Response formatting (add greeting, structure, emojis)
    4. User-friendly tone adjustment
    5. Adding helpful next steps
    
    Cost optimization: Reduces 2 LLM calls to 1 LLM call.
    """
    
    # Friendly emojis for different contexts
    EMOJI_MAP = {
        "greeting": "👋",
        "success": "✅",
        "tip": "💡",
        "warning": "⚠️",
        "academic": "🎓",
        "note": "📝",
    }
    
    def __init__(self, config: AgentConfig, agent_port):
        """
        Initialize the Response Formatter Agent.
        
        Args:
            config: Agent configuration containing model settings and parameters
            agent_port: Port for communicating with the underlying LLM
        """
        super().__init__(config, agent_port)
        
        # Extract agent-specific parameters from config
        params = getattr(config, 'parameters', {}) or {}
        self.preferred_tone = params.get('preferred_tone', 'friendly')
        self.max_response_length = params.get('max_response_length', 2000)
        self.include_next_steps = params.get('include_next_steps', True)
        self.max_emojis = params.get('max_emojis', 2)
    
    async def process(self, input_data: Dict[str, Any]) -> FormattedResponseResult:
        """
        Format and verify answer into user-friendly response.
        
        Args:
            input_data: Dictionary containing:
                - query: str - Original user query
                - answer: str - Generated answer to format
                - answer_confidence: Optional[float] - Answer generation confidence
                - context_documents: Optional[List[Dict]] - Source documents
                - user_context: Optional[Dict] - User information
        
        Returns:
            FormattedResponseResult containing formatted response with verification
        """
        query = input_data.get("query", "")
        answer = input_data.get("answer", "")
        answer_confidence = input_data.get("answer_confidence", 0.7)
        context_documents = input_data.get("context_documents", [])
        
        # Quick check for very short answers (direct formatting)
        if not answer or len(answer.strip()) < 10:
            return self._create_fallback_response(query, answer, "Answer too short")
        
        # Build the formatting prompt
        prompt = self._build_formatting_prompt(
            query, answer, answer_confidence, context_documents
        )
        
        # Get response from the agent (single LLM call)
        response = await self._make_agent_request(prompt)
        
        # Parse and return result
        return self._parse_response(response.content, query, answer)
    
    def _build_formatting_prompt(
        self,
        query: str,
        answer: str,
        confidence: float,
        context_documents: List[Dict[str, Any]]
    ) -> str:
        """Build prompt for combined verification and formatting."""
        # System prompt from config contains all instructions
        # Just provide the data to verify and format
        prompt_parts = [
            f"Query: {query}",
            f"\nAnswer to format:\n{answer}",
            f"\nOriginal confidence: {confidence:.2f}"
        ]
        
        if context_documents:
            prompt_parts.append(f"\nSources used: {len(context_documents)} documents")
            # Add brief source summaries
            for i, doc in enumerate(context_documents[:3], 1):
                title = doc.get("title", f"Document {i}")
                prompt_parts.append(f"  [{i}] {title}")
        
        return "\n".join(prompt_parts)
    
    def _parse_response(
        self, 
        response_content: str, 
        original_query: str,
        original_answer: str
    ) -> FormattedResponseResult:
        """Parse LLM response into FormattedResponseResult."""
        try:
            # Try to parse JSON response
            data = json.loads(response_content)
            return self._create_result_from_json(data, original_query, original_answer)
        except json.JSONDecodeError:
            # Fallback to rule-based formatting
            return self._create_fallback_response(original_query, original_answer, response_content)
    
    def _create_result_from_json(
        self, 
        data: Dict[str, Any], 
        original_query: str,
        original_answer: str
    ) -> FormattedResponseResult:
        """Create FormattedResponseResult from parsed JSON data."""
        quality_scores = data.get("quality_scores", {
            "accuracy": 7,
            "completeness": 7,
            "friendliness": 8
        })
        
        # Calculate overall score
        overall_score = (
            quality_scores.get("accuracy", 7) * 0.4 +
            quality_scores.get("completeness", 7) * 0.35 +
            quality_scores.get("friendliness", 8) * 0.25
        )
        
        final_response = data.get("final_response", original_answer)
        
        return FormattedResponseResult(
            final_response=final_response,
            needs_improvement=data.get("needs_improvement", False),
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            quality_scores=quality_scores,
            overall_score=overall_score,
            tone=data.get("tone", self.preferred_tone),
            includes_greeting=self._has_greeting(final_response),
            includes_next_steps=self._has_next_steps(final_response),
            confidence=0.85,
            metadata={"source": "llm_response"}
        )
    
    def _create_fallback_response(
        self, 
        query: str, 
        answer: str,
        context: str = ""
    ) -> FormattedResponseResult:
        """Create fallback response using rule-based formatting."""
        # Apply rule-based formatting
        formatted = self._apply_rule_based_formatting(query, answer)
        
        # Basic verification
        issues = self._identify_basic_issues(answer)
        suggestions = self._generate_suggestions(issues)
        
        # Calculate quality scores
        quality_scores = {
            "accuracy": max(5, 8 - len(issues)),
            "completeness": self._score_completeness(answer),
            "friendliness": self._score_friendliness(formatted)
        }
        
        overall_score = (
            quality_scores["accuracy"] * 0.4 +
            quality_scores["completeness"] * 0.35 +
            quality_scores["friendliness"] * 0.25
        )
        
        return FormattedResponseResult(
            final_response=formatted,
            needs_improvement=len(issues) > 1,
            issues=issues,
            suggestions=suggestions,
            quality_scores=quality_scores,
            overall_score=overall_score,
            tone=self._detect_tone(query),
            includes_greeting=self._has_greeting(formatted),
            includes_next_steps=self._has_next_steps(formatted),
            confidence=0.6,
            metadata={
                "fallback": True,
                "original_context": context[:200] if context else ""
            }
        )
    
    def _apply_rule_based_formatting(self, query: str, answer: str) -> str:
        """Apply rule-based formatting to answer."""
        if not answer or not answer.strip():
            return (
                "Chào bạn! 👋\n\n"
                "Xin lỗi, mình chưa có đủ thông tin để trả lời câu hỏi của bạn.\n\n"
                "Bạn có thể:\n"
                "- Cung cấp thêm chi tiết về câu hỏi\n"
                "- Liên hệ trực tiếp với UIT để được hỗ trợ\n\n"
                "Cần info gì khác thì hỏi mình nhé!"
            )
        
        formatted_parts = []
        
        # Add greeting based on query type
        greeting = self._generate_greeting(query)
        if greeting:
            formatted_parts.append(greeting)
        
        # Process main answer content
        processed_answer = self._process_answer_content(answer)
        formatted_parts.append(processed_answer)
        
        # Add closing/next steps
        if self.include_next_steps:
            closing = self._generate_closing(query)
            if closing:
                formatted_parts.append(closing)
        
        return "\n\n".join(formatted_parts)
    
    def _generate_greeting(self, query: str) -> str:
        """Generate appropriate greeting based on query."""
        query_lower = query.lower()
        
        # Social queries already have greetings
        if any(p in query_lower for p in ["xin chào", "hello", "hi", "chào"]):
            return "Chào bạn! 👋 Rất vui được hỗ trợ bạn."
        
        # Question queries
        return "Chào bạn! 👋"
    
    def _process_answer_content(self, answer: str) -> str:
        """Process and enhance answer content."""
        # Already well-formatted, return as is
        if "**" in answer or "- " in answer or "\n\n" in answer:
            return answer
        
        # Simple formatting enhancements
        processed = answer
        
        # Bold important numbers (Vietnamese currency, dates, etc.)
        import re
        
        # Bold currency amounts
        processed = re.sub(
            r'(\d{1,3}(?:\.\d{3})*(?:,\d+)?)\s*(VNĐ|đồng|triệu|nghìn)',
            r'**\1 \2**',
            processed
        )
        
        return processed
    
    def _generate_closing(self, query: str) -> str:
        """Generate closing statement with next steps."""
        query_lower = query.lower()
        
        if any(p in query_lower for p in ["học phí", "chi phí", "tiền"]):
            return "💡 Cần thêm info về thanh toán hay học bổng thì hỏi mình nhé!"
        
        if any(p in query_lower for p in ["đăng ký", "thủ tục", "quy trình"]):
            return "✅ Chúc bạn hoàn thành thủ tục thuận lợi! Có thắc mắc gì thêm cứ hỏi mình!"
        
        if any(p in query_lower for p in ["tốt nghiệp", "khóa luận", "điều kiện"]):
            return "🎓 Chúc bạn học tập tốt và tốt nghiệp thành công! Cần hỗ trợ thêm thì inbox mình nhé!"
        
        return "Cần thêm info gì khác thì hỏi mình nhé! 😊"
    
    def _identify_basic_issues(self, answer: str) -> List[str]:
        """Identify basic issues in the answer."""
        issues = []
        
        if len(answer.strip()) < 30:
            issues.append("Câu trả lời quá ngắn")
        
        # Check for generic/vague language
        vague_phrases = ["tùy thuộc vào", "có thể", "thường thì", "nói chung"]
        if sum(1 for p in vague_phrases if p in answer.lower()) > 3:
            issues.append("Câu trả lời thiếu thông tin cụ thể")
        
        return issues
    
    def _generate_suggestions(self, issues: List[str]) -> List[str]:
        """Generate improvement suggestions based on issues."""
        suggestions = []
        
        if any("ngắn" in issue for issue in issues):
            suggestions.append("Bổ sung thêm chi tiết và ví dụ")
        
        if any("cụ thể" in issue for issue in issues):
            suggestions.append("Thêm số liệu, ngày tháng cụ thể")
        
        return suggestions
    
    def _score_completeness(self, answer: str) -> int:
        """Score answer completeness (0-10)."""
        if not answer:
            return 0
        
        score = 5  # Base score
        
        # Length-based
        if len(answer) > 200:
            score += 2
        elif len(answer) > 100:
            score += 1
        
        # Structure indicators
        if any(p in answer.lower() for p in ["đầu tiên", "thứ hai", "cuối cùng"]):
            score += 1
        
        # Has examples
        if "ví dụ" in answer.lower():
            score += 1
        
        return min(10, score)
    
    def _score_friendliness(self, response: str) -> int:
        """Score response friendliness (0-10)."""
        score = 5  # Base score
        response_lower = response.lower()
        
        # Friendly indicators
        friendly_words = ["bạn", "mình", "hỗ trợ", "giúp", "nhé"]
        score += min(3, sum(1 for w in friendly_words if w in response_lower))
        
        # Has emojis
        if any(e in response for e in ["👋", "✅", "💡", "🎓", "😊"]):
            score += 1
        
        # Has greeting
        if self._has_greeting(response):
            score += 1
        
        return min(10, score)
    
    def _detect_tone(self, query: str) -> str:
        """Detect appropriate tone for response."""
        query_lower = query.lower()
        
        if any(p in query_lower for p in ["quy định", "policy", "chính sách"]):
            return "informative"
        
        if any(p in query_lower for p in ["hướng dẫn", "cách", "làm sao"]):
            return "helpful"
        
        return "friendly"
    
    def _has_greeting(self, text: str) -> bool:
        """Check if text has a greeting."""
        greetings = ["chào bạn", "xin chào", "hello", "hi"]
        return any(g in text.lower() for g in greetings)
    
    def _has_next_steps(self, text: str) -> bool:
        """Check if text has next steps or call to action."""
        next_step_indicators = ["hỏi mình", "liên hệ", "cần thêm", "nếu bạn"]
        return any(i in text.lower() for i in next_step_indicators)

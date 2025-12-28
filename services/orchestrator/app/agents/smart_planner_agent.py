"""
Smart Planner Agent implementation.

This agent combines the functionality of Planner and Query Rewriter agents
into a single optimized agent to reduce LLM calls.

Merged from:
- PlannerAgent: Query analysis, complexity scoring, execution planning
- QueryRewriterAgent: Query optimization, abbreviation expansion, UIT context

Enhanced with:
- Filter extraction: doc_types, faculties, years, subjects
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from ..agents.base import SpecializedAgent, AgentConfig, AgentType


@dataclass
class ExtractedFilters:
    """Filters extracted from the query context."""
    doc_types: List[str] = field(default_factory=list)  # e.g., ["syllabus", "regulation"]
    faculties: List[str] = field(default_factory=list)  # e.g., ["CNTT", "KHTN"]
    years: List[int] = field(default_factory=list)      # e.g., [2023, 2024]
    subjects: List[str] = field(default_factory=list)   # e.g., ["SE101", "CS201"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API payload."""
        result = {}
        if self.doc_types:
            result["doc_types"] = self.doc_types
        if self.faculties:
            result["faculties"] = self.faculties
        if self.years:
            result["years"] = self.years
        if self.subjects:
            result["subjects"] = self.subjects
        return result
    
    def is_empty(self) -> bool:
        """Check if no filters are set."""
        return not any([self.doc_types, self.faculties, self.years, self.subjects])


@dataclass
class SmartPlanResult:
    """Result from the Smart Planner agent (combined planning + query rewriting)."""
    # Planning fields
    query: str
    intent: str
    complexity: str  # "simple", "medium", "complex"
    complexity_score: float
    requires_rag: bool
    strategy: str  # "direct_response", "standard_rag", "advanced_rag"
    
    # Query rewriting fields
    rewritten_queries: List[str]
    search_terms: List[str]
    
    # RAG parameters
    top_k: int
    hybrid_search: bool
    reranking: bool
    
    # Metadata (required fields - no default)
    reasoning: str
    confidence: float
    metadata: Dict[str, Any]
    
    # Search source selection (optional fields with defaults - must come after required)
    use_knowledge_graph: bool = False  # Whether to use Knowledge Graph
    use_vector_search: bool = True     # Whether to use Vector Search
    
    # Graph Reasoning Type: "local", "global", "multi_hop"
    # - local: Simple 1-hop queries (prerequisites, department courses)
    # - global: Community-based summaries for comparative/overview questions
    # - multi_hop: Dynamic path exploration (e.g., "if I fail X, what courses are affected?")
    graph_query_type: str = "local"
    
    # Extracted filters for RAG search
    extracted_filters: Optional[ExtractedFilters] = None


class SmartPlannerAgent(SpecializedAgent):
    """
    Smart Planner Agent - Optimized combination of Planner + Query Rewriter.
    
    This agent performs in a single LLM call:
    1. Intent classification
    2. Complexity scoring (0-10 scale)
    3. Strategy determination (direct/standard_rag/advanced_rag)
    4. Query rewriting and optimization
    5. Search term extraction
    
    Cost optimization: Reduces 2 LLM calls to 1 LLM call.
    """
    
    # UIT-specific abbreviation mappings for fallback
    UIT_ABBREVIATIONS = {
        "hp": "học phần",
        "đkhp": "đăng ký học phần",
        "khmt": "khoa học máy tính",
        "cntt": "công nghệ thông tin",
        "httt": "hệ thống thông tin",
        "mmt": "mạng máy tính và truyền thông",
        "mmtt": "mạng máy tính và truyền thông",
        "sv": "sinh viên",
        "gv": "giảng viên",
        "đtbc": "điểm trung bình chung",
        "ctđt": "chương trình đào tạo",
        "uit": "đại học công nghệ thông tin",
        "đhqg": "đại học quốc gia",
        "hcm": "hồ chí minh",
    }
    
    # Complexity thresholds
    SIMPLE_MAX_SCORE = 3.5
    COMPLEX_MIN_SCORE = 6.5
    
    def __init__(self, config: AgentConfig, agent_port):
        """
        Initialize the Smart Planner Agent.
        
        Args:
            config: Agent configuration containing model settings and parameters
            agent_port: Port for communicating with the underlying LLM
        """
        super().__init__(config, agent_port)
        
        # Extract agent-specific parameters from config
        params = getattr(config, 'parameters', {}) or {}
        self.complexity_thresholds = params.get('complexity_thresholds', {
            'simple_max': 3.5,
            'complex_min': 6.5
        })
        self.default_top_k = params.get('default_top_k', 5)
        self.max_rewritten_queries = params.get('max_rewritten_queries', 3)
    
    async def process(self, input_data: Dict[str, Any]) -> SmartPlanResult:
        """
        Process user query and create combined plan with optimized queries.
        
        Args:
            input_data: Dictionary containing:
                - query: str - User query to analyze
                - context: Optional[Dict] - Additional context
                - user_profile: Optional[Dict] - User information
        
        Returns:
            SmartPlanResult containing planning + query rewriting results
        """
        query = input_data.get("query", "")
        
        # Quick check for very simple queries (no LLM needed)
        simple_result = self._check_simple_query(query)
        if simple_result:
            return simple_result
        
        # Build the analysis prompt
        prompt = self._build_analysis_prompt(query)
        
        # Get response from the agent (single LLM call)
        response = await self._make_agent_request(prompt)
        
        # Parse and return result
        return self._parse_response(response.content, query)
    
    def _check_simple_query(self, query: str) -> Optional[SmartPlanResult]:
        """
        Check if query is simple enough to handle without LLM.
        
        Returns SmartPlanResult for simple queries, None otherwise.
        """
        query_lower = query.lower().strip()
        
        # Identity questions - "Bạn là ai?"
        identity_patterns = [
            "bạn là ai", "bạn là gì", "mày là ai", "who are you", 
            "bạn tên gì", "tên bạn là gì", "bạn là chatbot gì"
        ]
        
        for pattern in identity_patterns:
            if pattern in query_lower:
                return SmartPlanResult(
                    query=query,
                    intent="social_greeting",
                    complexity="simple",
                    complexity_score=0.0,
                    requires_rag=False,
                    strategy="direct_response",
                    rewritten_queries=[],
                    search_terms=[],
                    top_k=0,
                    hybrid_search=False,
                    reranking=False,
                    reasoning="Identity question - direct response about chatbot",
                    confidence=1.0,
                    metadata={"rule_based": True, "pattern_matched": pattern}
                )
        
        # Social/greeting patterns
        social_patterns = [
            "xin chào", "hello", "hi", "chào", "cảm ơn", "thanks", "thank you",
            "tạm biệt", "bye", "ok", "được", "vâng", "dạ", "ừ", "oke"
        ]
        
        for pattern in social_patterns:
            if query_lower == pattern or query_lower.startswith(pattern + " "):
                return SmartPlanResult(
                    query=query,
                    intent="social_greeting",
                    complexity="simple",
                    complexity_score=0.0,
                    requires_rag=False,
                    strategy="direct_response",
                    rewritten_queries=[],
                    search_terms=[],
                    top_k=0,
                    hybrid_search=False,
                    reranking=False,
                    reasoning="Detected social/greeting query, no RAG needed",
                    confidence=1.0,
                    metadata={"rule_based": True, "pattern_matched": pattern}
                )
        
        return None
    
    def _build_analysis_prompt(self, query: str) -> str:
        """Build prompt for combined analysis and query rewriting."""
        # System prompt already contains all instructions from config
        # Just pass the user's query directly
        return query
    
    def _parse_response(self, response_content: str, original_query: str) -> SmartPlanResult:
        """Parse LLM response into SmartPlanResult."""
        try:
            # Try to parse JSON response
            data = json.loads(response_content)
            return self._create_result_from_json(data, original_query)
        except json.JSONDecodeError:
            # Fallback to rule-based analysis
            return self._create_fallback_result(original_query, response_content)
    
    def _create_result_from_json(self, data: Dict[str, Any], original_query: str) -> SmartPlanResult:
        """Create SmartPlanResult from parsed JSON data with optimized reranking strategy."""
        complexity_score = data.get("complexity_score", 5.0)
        complexity = data.get("complexity", self._score_to_complexity(complexity_score))
        requires_rag = data.get("requires_rag", True)
        strategy = data.get("strategy", "standard_rag")
        intent = data.get("intent", "informational")
        
        # Determine RAG parameters based on complexity with optimized reranking strategy
        # OPTIMIZATION: Only use reranking for high complexity (score > 7.0) to reduce latency
        if complexity == "simple" or not requires_rag:
            top_k = 0
            hybrid_search = False
            reranking = False
        elif complexity == "complex" and complexity_score > 7.0:
            # High complexity: Use aggressive search with reranking
            top_k = data.get("top_k", 10)
            hybrid_search = data.get("hybrid_search", True)
            reranking = True  # Only enable for very complex queries
        elif complexity == "complex":
            # Medium-high complexity: Use hybrid but skip reranking
            top_k = data.get("top_k", 8)
            hybrid_search = data.get("hybrid_search", True)
            reranking = False  # Skip reranking for faster response
        else:  # medium
            # Medium complexity: Fast vector-only search, no reranking
            top_k = data.get("top_k", 5)
            hybrid_search = False  # Vector-only for speed
            reranking = False  # No reranking for medium complexity
        
        # Determine search sources based on complexity, strategy, and query content
        # Priority 1: Read use_knowledge_graph from LLM response (if provided)
        # Priority 2: Fallback to rule-based detection
        # Use Knowledge Graph for:
        # 1. LLM explicitly set use_knowledge_graph=true in response
        # 2. Complex queries
        # 3. Advanced RAG strategy
        # 4. Queries about relationships between concepts/articles/regulations
        needs_relationship_search = self._needs_knowledge_graph(original_query)
        
        # CRITICAL FIX: Always use Knowledge Graph when RAG is needed
        # This ensures the LLM gets modification context (replaced/amended articles)
        # Knowledge Graph contains relationship data that vector search misses
        use_knowledge_graph = data.get("use_knowledge_graph", None)
        if use_knowledge_graph is None:
            # NEW: Always enable KG when RAG is needed for modification awareness
            # Previously: Only enabled for complex queries, advanced_rag, relationship queries
            # Now: Always enabled with RAG to provide article modification context
            use_knowledge_graph = requires_rag  # Simple: if we need RAG, we also need KG
        
        # Always use vector search for RAG queries (primary source)
        use_vector_search = requires_rag
        
        # Determine graph query type for advanced reasoning
        # Priority 1: Read from LLM response
        # Priority 2: Fallback to rule-based detection
        graph_query_type = data.get("graph_query_type", None)
        if graph_query_type is None and use_knowledge_graph:
            graph_query_type = self._determine_graph_query_type(original_query)
        elif graph_query_type is None:
            graph_query_type = "local"
        
        # Extract filters from query context
        extracted_filters = self._extract_filters_from_query(original_query) if requires_rag else None
        
        return SmartPlanResult(
            query=original_query,
            intent=intent,
            complexity=complexity,
            complexity_score=complexity_score,
            requires_rag=requires_rag,
            strategy=strategy,
            rewritten_queries=data.get("rewritten_queries", [original_query]),
            search_terms=data.get("search_terms", self._extract_keywords(original_query)),
            top_k=top_k,
            hybrid_search=hybrid_search,
            reranking=reranking,
            use_knowledge_graph=use_knowledge_graph,
            use_vector_search=use_vector_search,
            graph_query_type=graph_query_type,
            extracted_filters=extracted_filters,
            reasoning=data.get("reasoning", ""),
            confidence=0.85,
            metadata={
                "source": "llm_response", 
                "kg_reason": "relationship_query" if needs_relationship_search else None,
                "graph_reasoning_type": graph_query_type
            }
        )
    
    def _create_fallback_result(self, query: str, response_content: str) -> SmartPlanResult:
        """Create fallback result using rule-based analysis."""
        # Estimate complexity
        complexity_score = self._estimate_complexity_score(query)
        complexity = self._score_to_complexity(complexity_score)
        
        # Determine if RAG is needed
        requires_rag = complexity != "simple"
        
        # Generate rewritten queries
        rewritten_queries = self._apply_rule_based_rewriting(query) if requires_rag else []
        
        # Extract search terms
        search_terms = self._extract_keywords(query) if requires_rag else []
        
        # Determine RAG parameters with optimized reranking strategy
        if complexity == "simple":
            top_k, hybrid_search, reranking = 0, False, False
            strategy = "direct_response"
        elif complexity == "complex" and complexity_score > 7.0:
            # High complexity: Use reranking for best accuracy
            top_k, hybrid_search, reranking = 10, True, True
            strategy = "advanced_rag"
        elif complexity == "complex":
            # Medium-high complexity: Skip reranking for speed
            top_k, hybrid_search, reranking = 8, True, False
            strategy = "advanced_rag"
        else:
            # Medium complexity: Fast vector-only search
            top_k, hybrid_search, reranking = 5, False, False
            strategy = "standard_rag"
        
        # Determine search sources based on complexity and relationship queries
        intent = self._detect_intent(query)
        needs_relationship_search = self._needs_knowledge_graph(query)
        use_knowledge_graph = (
            requires_rag and 
            (complexity == "complex" or needs_relationship_search or intent == "comparative")
        )
        use_vector_search = requires_rag
        
        # Determine graph query type for advanced reasoning
        graph_query_type = self._determine_graph_query_type(query) if use_knowledge_graph else "local"
        
        # Extract filters from query context
        extracted_filters = self._extract_filters_from_query(query) if requires_rag else None
        
        return SmartPlanResult(
            query=query,
            intent=intent,
            complexity=complexity,
            complexity_score=complexity_score,
            requires_rag=requires_rag,
            strategy=strategy,
            rewritten_queries=rewritten_queries,
            search_terms=search_terms,
            top_k=top_k,
            hybrid_search=hybrid_search,
            reranking=reranking,
            use_knowledge_graph=use_knowledge_graph,
            use_vector_search=use_vector_search,
            graph_query_type=graph_query_type,
            extracted_filters=extracted_filters,
            reasoning="Fallback to rule-based analysis",
            confidence=0.6,
            metadata={
                "fallback": True,
                "original_response": response_content[:200],
                "kg_reason": "relationship_query" if needs_relationship_search else None,
                "graph_reasoning_type": graph_query_type
            }
        )
    
    def _score_to_complexity(self, score: float) -> str:
        """Convert complexity score to label."""
        if score <= self.SIMPLE_MAX_SCORE:
            return "simple"
        elif score >= self.COMPLEX_MIN_SCORE:
            return "complex"
        return "medium"
    
    def _estimate_complexity_score(self, query: str) -> float:
        """Estimate complexity score using heuristics."""
        score = 5.0  # Default medium
        query_lower = query.lower()
        
        # Simple indicators (reduce score)
        simple_patterns = ["xin chào", "hello", "cảm ơn", "tạm biệt", "ok"]
        if any(p in query_lower for p in simple_patterns):
            return 1.0
        
        # Short queries tend to be simpler
        if len(query) < 20:
            score -= 2.0
        elif len(query) < 40:
            score -= 1.0
        
        # Complex indicators (increase score)
        complex_patterns = [
            "so sánh", "phân tích", "đánh giá", "quy trình chi tiết",
            "hướng dẫn", "các bước", "làm thế nào", "khác biệt"
        ]
        if any(p in query_lower for p in complex_patterns):
            score += 2.5
        
        # Multiple question indicators
        if query.count("?") > 1:
            score += 1.0
        
        # Long queries tend to be more complex
        if len(query) > 100:
            score += 1.5
        elif len(query) > 60:
            score += 0.5
        
        return max(0, min(10, score))  # Clamp to 0-10
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent using rule-based approach."""
        query_lower = query.lower()
        
        # Social
        if any(p in query_lower for p in ["xin chào", "hello", "cảm ơn", "tạm biệt"]):
            return "social_greeting"
        
        # Comparative
        if any(p in query_lower for p in ["so sánh", "khác biệt", "giống nhau", "vs"]):
            return "comparative"
        
        # Procedural
        if any(p in query_lower for p in ["cách", "làm sao", "thế nào", "quy trình", "hướng dẫn"]):
            return "procedural"
        
        # Informational (default)
        return "informational"
    
    def _needs_knowledge_graph(self, query: str) -> bool:
        """
        Determine if the query would benefit from Knowledge Graph search.
        
        Knowledge Graph is particularly useful for:
        - Questions about specific articles/regulations (structured data in KG)
        - Questions about conditions, requirements in regulations
        - Questions about relationships between articles/regulations
        - Questions involving multiple connected concepts  
        - Questions that require traversing linked information
        """
        import re
        query_lower = query.lower()
        
        # Relationship indicators - questions about connections between things
        relationship_patterns = [
            "mối quan hệ", "quan hệ", "liên quan", "liên kết",
            "kết nối", "ảnh hưởng", "tác động", "phụ thuộc",
            "dẫn đến", "gây ra", "bắt nguồn từ"
        ]
        
        # Article/regulation reference patterns  
        regulation_patterns = [
            "khoản", "mục", "chương", "quy chế", "nghị định", "thông tư"
        ]
        
        # Regulation-related question patterns - questions about conditions/requirements
        regulation_questions = [
            "điều kiện", "yêu cầu", "quy định", "thủ tục", "hồ sơ",
            "chuyển ngành", "chuyển trường", "chuyển chương trình",
            "bảo lưu", "thôi học", "tốt nghiệp", "xét tốt nghiệp",
            "khen thưởng", "kỷ luật", "học bổng"
        ]
        
        # Check for relationship keywords
        has_relationship = any(p in query_lower for p in relationship_patterns)
        
        # Check for regulation/article references (excluding simple "điều" and "quy định")
        has_regulation = any(p in query_lower for p in regulation_patterns)
        
        # Check for regulation-related questions (conditions, requirements)
        has_regulation_question = any(p in query_lower for p in regulation_questions)
        
        # Use regex to properly count distinct article references
        # Pattern matches "điều X" where X is a number (may have words in between)
        article_pattern = r'điều\s+(\d+)'  # More flexible: requires space(s) before number
        article_matches = re.findall(article_pattern, query_lower)
        unique_articles = set(article_matches)
        has_single_article = len(unique_articles) >= 1  # Changed: even 1 article should use KG
        has_multiple_articles = len(unique_articles) >= 2
        
        # Check for comparative patterns with regulations
        comparative_regulation = (
            (has_regulation or len(unique_articles) >= 1) and 
            any(p in query_lower for p in ["so sánh", "khác", "giống", "với"])
        )
        
        # Enable KG if:
        # 1. Query explicitly asks about relationships
        # 2. Query references ANY article (even single article - KG has structured data)
        # 3. Query compares regulations
        # 4. Query references other regulation patterns (khoản, mục, chương, etc.)
        # 5. Query asks about conditions/requirements (likely from regulations in KG)
        return (
            has_relationship or
            has_single_article or  # Use KG for even single article queries
            has_multiple_articles or
            comparative_regulation or
            has_regulation or  # Use KG for regulation patterns
            has_regulation_question  # NEW: Use KG for condition/requirement questions
        )

    def _determine_graph_query_type(self, query: str) -> str:
        """
        Determine the type of graph query needed for reasoning.
        
        Types:
        - "local": Simple 1-hop queries (prerequisites, department courses)
        - "global": Community-based summaries for comparative/overview questions
        - "multi_hop": Dynamic path exploration with impact analysis
        
        Examples:
        - LOCAL: "Môn IT003 cần học môn gì trước?" → prerequisite lookup
        - GLOBAL: "So sánh cấu trúc chương trình CNTT và KHMT?" → community summaries
        - MULTI_HOP: "Nếu tôi rớt IT001 thì tôi sẽ bị trễ môn đồ án nào?" → chain reasoning
        """
        query_lower = query.lower()
        
        # === MULTI_HOP PATTERNS ===
        # Questions about impact/consequence/chain effects
        multi_hop_patterns = [
            # Conditional impact
            r'nếu.*(rớt|trượt|không qua|fail).*thì',
            r'nếu.*(không học|bỏ qua|skip).*thì',
            r'(rớt|trượt).*ảnh hưởng',
            r'(rớt|trượt).*bị trễ',
            # Chain questions
            r'chuỗi.*(môn|học phần)',
            r'từ.*(cơ sở|nền tảng).*đến.*(chuyên ngành|nâng cao)',
            r'đường đi.*học',
            r'lộ trình.*học',
            # Impact analysis
            r'ảnh hưởng.*như thế nào.*tốt nghiệp',
            r'tác động.*đến.*năm (cuối|\d)',
        ]
        
        for pattern in multi_hop_patterns:
            if re.search(pattern, query_lower):
                return "multi_hop"
        
        # === LOCAL PATTERNS (CHECK FIRST - for relationship scanning) ===
        # These should use LOCAL with ReAct to call scan_relationships tool
        relationship_scan_patterns = [
            r'liệt kê.*(có|với).*(quan hệ|mối quan hệ).*(yeu_cau|quy_dinh|điều kiện)',
            r'(các điều|điều khoản).*(có|với).*(quan hệ|relationship)',
            r'scan.*relationship',
            r'tìm.*(các cặp|cặp).*(có|với).*quan hệ',
        ]
        
        for pattern in relationship_scan_patterns:
            if re.search(pattern, query_lower):
                return "local"  # Use LOCAL with ReAct for scan_relationships tool
        
        # === GLOBAL PATTERNS ===
        # Comparative, overview, summary questions (but NOT relationship scans)
        global_patterns = [
            # Comparison
            r'so sánh.*(chương trình|ngành|khoa)',
            r'khác biệt.*(giữa|của).*(chương trình|ngành|khoa)',
            r'(cntt|khmt|httt|mmt).*(vs|so với|và).*(cntt|khmt|httt|mmt)',
            # Overview/Summary (exclude relationship queries)
            r'tóm tắt.*(quy định|chương trình|môn học)',
            r'tổng quan.*(về|của)',
            r'liệt kê tất cả.*(môn|chương trình)',  # More specific
            r'có bao nhiêu (môn|quy định|điều)',
            # Categorization
            r'phân loại.*(môn|quy định)',
            r'nhóm.*(môn học|quy định)',
        ]
        
        for pattern in global_patterns:
            if re.search(pattern, query_lower):
                return "global"
        
        # === LOCAL (DEFAULT) ===
        # Simple lookup patterns
        local_patterns = [
            # Prerequisites
            r'(tiên quyết|học trước|cần học)',
            r'môn.*(trước|sau)',
            # Department/Program
            r'(thuộc|của) khoa',
            r'môn.*(bắt buộc|tự chọn)',
            # Single entity
            r'^(cho biết|thông tin về)',
        ]
        
        for pattern in local_patterns:
            if re.search(pattern, query_lower):
                return "local"
        
        # Default to local for simple knowledge graph queries
        return "local"

    def _apply_rule_based_rewriting(self, query: str) -> List[str]:
        """Apply rule-based query rewriting."""
        query_lower = query.lower()
        rewritten = []
        
        # Expand abbreviations
        expanded = query_lower
        for abbr, full in self.UIT_ABBREVIATIONS.items():
            if abbr in expanded:
                expanded = expanded.replace(abbr, full)
        
        if expanded != query_lower:
            rewritten.append(expanded)
        
        # Add UIT context if not present
        if "uit" not in query_lower and "đại học công nghệ" not in query_lower:
            rewritten.append(f"{query} tại UIT")
        
        # Original query as fallback
        if query not in rewritten:
            rewritten.insert(0, query)
        
        return rewritten[:self.max_rewritten_queries]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Vietnamese stop words
        stop_words = {
            "là", "của", "và", "có", "trong", "với", "để", "về", "tại", "từ",
            "này", "đó", "những", "các", "một", "như", "được", "sẽ", "đã",
            "đang", "thì", "nếu", "khi", "mà", "hay", "hoặc", "gì", "nào"
        }
        
        # Important UIT terms
        important_terms = {
            "uit", "đhqg", "học phần", "môn học", "tín chỉ", "học phí",
            "tuyển sinh", "khoa", "ngành", "quy định", "thủ tục", "đăng ký",
            "tốt nghiệp", "điều kiện", "điểm", "sinh viên", "giảng viên"
        }
        
        # Tokenize and filter
        words = query.lower().replace(",", " ").replace(".", " ").split()
        keywords = []
        
        for word in words:
            word = word.strip("?!.,;:")
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        return keywords[:10]

    def _extract_filters_from_query(self, query: str) -> ExtractedFilters:
        """
        Extract search filters from the query context.
        
        Detects:
        - doc_types: quy chế, đề cương, thông báo, etc.
        - faculties: CNTT, KHTN, MMT, etc.
        - years: 2023, 2024, etc.
        - subjects: SE101, CS201, etc.
        
        Args:
            query: User query to extract filters from
            
        Returns:
            ExtractedFilters with detected values
        """
        query_lower = query.lower()
        filters = ExtractedFilters()
        
        # === DOC_TYPES DETECTION ===
        # TEMPORARILY DISABLED: Current data has doc_type="Quy chế Đào tạo" which doesn't match
        # the filter values like "regulation". Need to fix data indexing or add mapping layer.
        # TODO: Re-enable when doc_type values are normalized in the data.
        # doc_type_patterns = {
        #     "regulation": ["quy chế", "quy định", "điều lệ", "nội quy"],
        #     "syllabus": ["đề cương", "chương trình đào tạo", "ctđt", "curriculum"],
        #     "announcement": ["thông báo", "công văn", "hướng dẫn"],
        #     "form": ["biểu mẫu", "đơn", "form"],
        #     "handbook": ["sổ tay", "cẩm nang", "hướng dẫn sinh viên"],
        # }
        # 
        # for doc_type, patterns in doc_type_patterns.items():
        #     if any(p in query_lower for p in patterns):
        #         filters.doc_types.append(doc_type)
        
        # === FACULTIES DETECTION ===
        # Note: Using word boundaries to avoid false positives (e.g., "UIT" matching "it")
        faculty_patterns = {
            "CNTT": ["công nghệ thông tin", "cntt", "khoa cntt"],
            "KHMT": ["khoa học máy tính", "khmt", "computer science"],
            "HTTT": ["hệ thống thông tin", "httt", "information systems"],
            "MMT": ["mạng máy tính", "mmt", "mmtt", "truyền thông", "network"],
            "KTMT": ["kỹ thuật máy tính", "ktmt", "computer engineering"],
            "KHTN": ["khoa học tự nhiên", "khtn"],
            "CTDA": ["công trình đa âm", "ctda"],
        }
        
        for faculty, patterns in faculty_patterns.items():
            if any(p in query_lower for p in patterns):
                filters.faculties.append(faculty)
        
        # === YEARS DETECTION ===
        # Extract years from patterns like "năm 2023", "2024", "khóa 2023"
        year_patterns = [
            r'năm\s*(\d{4})',
            r'khóa\s*(\d{4})',
            r'niên khóa\s*(\d{4})',
            r'(\d{4})\s*[-–]\s*\d{4}',  # Range like 2023-2024
            r'\b(20\d{2})\b'  # Standalone year 2000-2099
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, query_lower)
            for match in matches:
                try:
                    year = int(match)
                    if 2000 <= year <= 2100 and year not in filters.years:
                        filters.years.append(year)
                except ValueError:
                    continue
        
        # === SUBJECTS DETECTION ===
        # Pattern for subject codes like SE101, CS201, IT001
        subject_pattern = r'\b([A-Z]{2,4}\d{3})\b'
        subject_matches = re.findall(subject_pattern, query.upper())
        
        for match in subject_matches:
            if match not in filters.subjects:
                filters.subjects.append(match)
        
        # Also check for common subject name patterns
        subject_name_patterns = {
            "IT001": ["nhập môn lập trình", "intro to programming"],
            "IT002": ["lập trình hướng đối tượng", "oop"],
            "IT003": ["cấu trúc dữ liệu", "data structures"],
            "SE100": ["nhập môn công nghệ phần mềm"],
            "SE101": ["công nghệ phần mềm"],
        }
        
        for subject_code, patterns in subject_name_patterns.items():
            if any(p in query_lower for p in patterns):
                if subject_code not in filters.subjects:
                    filters.subjects.append(subject_code)
        
        return filters

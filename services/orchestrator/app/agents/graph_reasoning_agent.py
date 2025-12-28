"""
Graph Reasoning Agent - Advanced Graph-based RAG with Dynamic Reasoning

This agent extends beyond simple graph lookup to provide:
1. LOCAL: Pattern-based 1-hop queries (existing)
2. GLOBAL: Community-based summaries for comparative/overview questions
3. MULTI-HOP: Dynamic path exploration with LLM-guided reasoning

Addresses the gap between:
- "Graph Lookup" (tra cứu theo mẫu cứng)
- "Graph Reasoning" (suy luận động)

Key improvements:
- From Static patterns to Dynamic traversal
- From Local (single node) to Global (community summaries)
- From 1-hop to Multi-hop reasoning chains

ReAct Framework (Reasoning + Acting):
- Tools: Graph operations that LLM can call
- Loop: Thought → Action → Observation → Repeat until done
"""

import json
import logging
import re
import sys
import os
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

# Add rag_services to path for importing NodeCategory
_rag_services_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'rag_services'))
if _rag_services_path not in sys.path:
    sys.path.insert(0, _rag_services_path)

# Import NodeCategory from rag_services
try:
    from core.domain.graph_models import NodeCategory
except ImportError:
    # Fallback: Define locally if import fails
    class NodeCategory(str, Enum):
        MON_HOC = "MON_HOC"
        QUY_DINH = "QUY_DINH"
        DIEU_KIEN = "DIEU_KIEN"
        KHOA = "KHOA"
        NGANH = "NGANH"
        CHUONG_TRINH_DAO_TAO = "CHUONG_TRINH_DAO_TAO"

logger = logging.getLogger(__name__)


# ========== REACT TOOL DEFINITIONS ==========

@dataclass
class ToolResult:
    """Result from a tool execution."""
    success: bool
    data: Any = None
    error: str = ""
    
    def to_observation(self) -> str:
        """Convert to observation string for LLM."""
        if self.success:
            if isinstance(self.data, list):
                if len(self.data) == 0:
                    return "Không tìm thấy kết quả."
                return f"Tìm thấy {len(self.data)} kết quả:\n" + "\n".join(
                    f"  - {self._format_item(item)}" for item in self.data[:10]
                )
            elif isinstance(self.data, dict):
                return f"Kết quả: {json.dumps(self.data, ensure_ascii=False, indent=2)}"
            else:
                return str(self.data)
        else:
            return f"Lỗi: {self.error}"
    
    def _format_item(self, item: Any) -> str:
        """Format a single item for display."""
        if isinstance(item, dict):
            # Support both MON_HOC nodes and Article nodes
            name = (item.get("name") or item.get("title") or 
                   item.get("ma_mon") or item.get("ten_mon") or 
                   item.get("article_id", "Unknown"))
            item_type = item.get("type") or item.get("node_type", "")
            return f"[{item_type}] {name}"
        return str(item)


@dataclass
class Tool:
    """Definition of a tool that LLM can call."""
    name: str
    description: str
    parameters: Dict[str, str]  # param_name -> description
    examples: List[str] = field(default_factory=list)
    
    def to_prompt_string(self) -> str:
        """Convert tool to prompt string for LLM."""
        params_str = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        examples_str = "\n".join(f"    Ví dụ: {ex}" for ex in self.examples)
        return f"""- {self.name}({params_str})
    Mô tả: {self.description}
{examples_str}"""


class GraphQueryType(str, Enum):
    """Types of graph queries based on reasoning complexity."""
    LOCAL = "local"          # Simple 1-hop lookup (existing patterns)
    GLOBAL = "global"        # Community-based global reasoning
    MULTI_HOP = "multi_hop"  # Dynamic multi-hop path exploration


@dataclass
class GraphReasoningResult:
    """Result from graph reasoning."""
    query_type: GraphQueryType
    query: str
    
    # Retrieved graph data
    nodes: List[Dict[str, Any]] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    paths: List[Dict[str, Any]] = field(default_factory=list)
    
    # For global reasoning
    community_summaries: List[str] = field(default_factory=list)
    
    # Reasoning chain (for multi-hop)
    reasoning_steps: List[str] = field(default_factory=list)
    
    # Final synthesized context for AnswerAgent
    synthesized_context: str = ""
    
    # Metadata
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_context_string(self) -> str:
        """Convert result to context string for AnswerAgent."""
        if self.synthesized_context:
            return self.synthesized_context
        
        parts = []
        
        # Add reasoning steps if available
        if self.reasoning_steps:
            parts.append("=== Graph Reasoning Steps ===")
            for i, step in enumerate(self.reasoning_steps, 1):
                parts.append(f"{i}. {step}")
            parts.append("")
        
        # Add community summaries for global queries
        if self.community_summaries:
            parts.append("=== Relevant Community Summaries ===")
            for summary in self.community_summaries:
                parts.append(f"• {summary}")
            parts.append("")
        
        # Add paths for multi-hop
        if self.paths:
            parts.append("=== Discovered Paths ===")
            for path in self.paths:
                path_str = " → ".join(path.get("node_names", []))
                parts.append(f"• {path_str}")
            parts.append("")
        
        # Add nodes
        if self.nodes:
            parts.append("=== Related Nodes ===")
            for node in self.nodes[:10]:  # Limit to 10
                # Support both MON_HOC nodes and Article nodes
                name = (node.get("name") or node.get("title") or 
                       node.get("ten_mon") or node.get("ma_mon") or 
                       node.get("article_id", "Unknown"))
                node_type = node.get("type", "Node")
                
                # FIX: Include content/full_text in context!
                content = node.get("content") or node.get("full_text") or node.get("noi_dung") or node.get("text")
                
                if content:
                    # Add title and content
                    # Note: If article was amended, the content is already replaced with new version
                    parts.append(f"• [{node_type}] {name}")
                    parts.append(f"  Content: {content}")
                    parts.append("")  # Add blank line for readability
                else:
                    # Fallback: just title if no content
                    parts.append(f"• [{node_type}] {name}")
        
        return "\n".join(parts) if parts else "No graph context found."


class GraphReasoningAgent:
    """
    Agent for advanced graph-based reasoning with ReAct framework.
    
    Capabilities:
    1. LOCAL: Pattern-based queries (prerequisites, department courses, etc.)
    2. GLOBAL: Community detection summaries for overview/comparative questions
    3. MULTI-HOP: Dynamic path exploration with LLM guidance using ReAct
    
    ReAct (Reasoning + Acting):
    - LLM decides which tools to call based on reasoning
    - Loop: Thought → Action → Observation → Repeat
    - Supports complex queries that don't match static patterns
    
    Example queries handled:
    - LOCAL: "Môn IT003 cần học môn gì trước?" 
    - GLOBAL: "So sánh cấu trúc chương trình CNTT và KHMT?"
    - MULTI-HOP: "Nếu tôi rớt IT001 thì tôi sẽ bị trễ môn đồ án nào?"
    """
    
    # ReAct Tools definition
    REACT_TOOLS: List[Tool] = [
        Tool(
            name="search_articles",
            description="Tìm kiếm các điều khoản (Article) theo từ khóa",
            parameters={"keywords": "danh sách từ khóa, phân cách bởi dấu phẩy"},
            examples=['search_articles("đăng ký, học phần")', 'search_articles("tín chỉ, tốt nghiệp")']
        ),
        Tool(
            name="search_entities",
            description="Tìm kiếm các thực thể (Entity) như sinh viên, học phần, điểm số",
            parameters={"keywords": "danh sách từ khóa, phân cách bởi dấu phẩy"},
            examples=['search_entities("sinh viên")', 'search_entities("điểm trung bình")']
        ),
        Tool(
            name="get_article",
            description="Lấy nội dung chi tiết của một Điều (Article) theo số điều",
            parameters={"article_number": "số điều (ví dụ: 14 cho Điều 14)"},
            examples=['get_article(14)', 'get_article(25)']
        ),
        Tool(
            name="get_related_articles",
            description="Tìm các điều khoản liên quan đến một thực thể",
            parameters={"entity_name": "tên thực thể cần tìm điều khoản liên quan"},
            examples=['get_related_articles("sinh viên")', 'get_related_articles("học phần")']
        ),
        Tool(
            name="get_communities",
            description="Lấy tóm tắt các Community (nhóm điều khoản) trong graph",
            parameters={},
            examples=['get_communities()']
        ),
        Tool(
            name="find_article_path",
            description="Tìm đường đi giữa 2 điều khoản (Article)",
            parameters={
                "start_article": "số điều bắt đầu",
                "end_article": "số điều kết thúc"
            },
            examples=['find_article_path(10, 25)', 'find_article_path(14, 30)']
        ),
        Tool(
            name="find_prerequisites",
            description="Tìm chuỗi môn học tiên quyết của một môn (course code)",
            parameters={"course_code": "mã môn học (ví dụ: IT001, SE104)"},
            examples=['find_prerequisites("IT001")', 'find_prerequisites("SE104")']
        ),
        Tool(
            name="find_dependents",
            description="Tìm các môn học phụ thuộc vào môn này (môn nào cần học môn này trước)",
            parameters={"course_code": "mã môn học"},
            examples=['find_dependents("IT001")', 'find_dependents("MA001")']
        ),
        Tool(
            name="scan_relationships",
            description="Liệt kê các cặp node có mối quan hệ cụ thể (ví dụ: YEU_CAU, QUY_DINH_DIEU_KIEN)",
            parameters={"rel_types": "danh sách loại quan hệ, phân cách dấu phẩy"},
            examples=['scan_relationships("YEU_CAU")', 'scan_relationships("YEU_CAU, QUY_DINH_DIEU_KIEN")']
        ),
        Tool(
            name="check_amendments",
            description="Kiểm tra xem một điều khoản đã bị sửa đổi/thay thế chưa và lấy nội dung mới nhất. QUAN TRỌNG: Luôn gọi tool này khi tìm thấy điều khoản để đảm bảo trả lời với thông tin mới nhất.",
            parameters={"article_title": "tiêu đề điều khoản cần kiểm tra (ví dụ: 'Điều 14')"},
            examples=['check_amendments("Điều 14")', 'check_amendments("Điều 4")']
        ),
        Tool(
            name="done",
            description="Kết thúc reasoning khi đã có đủ thông tin để trả lời",
            parameters={"summary": "tóm tắt kết quả tìm được"},
            examples=['done("Tìm thấy 5 điều khoản liên quan đến đăng ký học phần")']
        ),
    ]
    
    # ReAct prompt template
    REACT_SYSTEM_PROMPT = """Bạn là một agent chuyên suy luận trên đồ thị tri thức (Knowledge Graph) về quy chế đào tạo của trường đại học.

Nhiệm vụ: Trả lời câu hỏi bằng cách suy luận và gọi các công cụ (tools) để thu thập thông tin từ graph.

Các công cụ có sẵn:
{tools_description}

Quy tắc:
1. Bắt đầu bằng "Thought:" - suy nghĩ cần làm gì tiếp theo
2. Tiếp theo là "Action:" - gọi MỘT công cụ với cú pháp đúng
3. Sau khi nhận "Observation:" - đánh giá kết quả và quyết định bước tiếp theo
4. Lặp lại cho đến khi có đủ thông tin, sau đó gọi done() với tóm tắt

Ví dụ:
User: Điều 14 quy định về vấn đề gì?
Thought: Cần lấy nội dung chi tiết của Điều 14
Action: get_article(14)
Observation: Kết quả: {{title: "Đăng ký học phần", content: "..."}}
Thought: Đã có đủ thông tin về Điều 14
Action: done("Điều 14 quy định về đăng ký học phần")

QUAN TRỌNG:
- Chỉ gọi MỘT action mỗi lần
- Sử dụng đúng cú pháp: tool_name(param1, param2)
- Luôn kết thúc bằng done() khi đã đủ thông tin
"""
    
    def __init__(self, graph_adapter, llm_port=None, react_model: Optional[str] = None):
        """
        Initialize GraphReasoningAgent.
        
        Args:
            graph_adapter: Neo4j adapter for graph operations
            llm_port: Optional LLM port for dynamic ReAct reasoning
            react_model: Optional model name for ReAct LLM calls (None = use default)
        """
        self.graph_adapter = graph_adapter
        self.llm_port = llm_port
        self.react_model = react_model  # Model to use for ReAct reasoning
        
        # Max depth for multi-hop reasoning
        self.max_hop_depth = 5
        
        # Community retrieval settings
        self.max_communities = 3
        
        # ReAct settings - OPTIMIZED: reduced from 8 to 3 for faster response
        # Each iteration is ~1.7s LLM call, so 3 iterations = ~5s vs 8 iterations = ~14s
        self.max_react_iterations = 3
        
        # Build tools mapping
        self._tools_map: Dict[str, Callable] = {}
        self._init_tools()
        
        if react_model:
            logger.info(f"GraphReasoningAgent initialized with model: {react_model}")
        else:
            logger.info("GraphReasoningAgent initialized with default model")
        
        logger.info("GraphReasoningAgent initialized with ReAct framework")
    
    # ========== REACT FRAMEWORK ==========
    
    def _init_tools(self):
        """Initialize the tools mapping for ReAct."""
        self._tools_map = {
            "search_articles": self._tool_search_articles,
            "search_entities": self._tool_search_entities,
            "get_article": self._tool_get_article,
            "get_related_articles": self._tool_get_related_articles,
            "get_communities": self._tool_get_communities,
            "find_article_path": self._tool_find_article_path,
            "find_prerequisites": self._tool_find_prerequisites,
            "find_dependents": self._tool_find_dependents,
            "scan_relationships": self._tool_scan_relationships,
            "check_amendments": self._tool_check_amendments,
            "done": self._tool_done,
        }
    
    def _get_tools_description(self) -> str:
        """Get formatted tools description for LLM prompt."""
        return "\n".join(tool.to_prompt_string() for tool in self.REACT_TOOLS)
    
    async def _react_loop(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> GraphReasoningResult:
        """
        Execute the ReAct (Reasoning + Acting) loop.
        
        This allows LLM to dynamically decide which tools to call
        based on step-by-step reasoning.
        
        Args:
            query: User query
            context: Additional context
            
        Returns:
            GraphReasoningResult with collected information
        """
        result = GraphReasoningResult(
            query_type=GraphQueryType.MULTI_HOP,
            query=query
        )
        
        # Check if LLM is available
        if not self.llm_port:
            logger.warning("LLM not available for ReAct, falling back to pattern-based")
            return await self._pattern_based_multi_hop(query, context)
        
        # Build conversation history for ReAct
        conversation = []
        collected_data = {
            "nodes": [],
            "paths": [],
            "communities": [],
            "articles": [],
        }
        
        # System prompt with tools
        system_prompt = self.REACT_SYSTEM_PROMPT.format(
            tools_description=self._get_tools_description()
        )
        
        # Initial user message
        user_message = f"Câu hỏi: {query}"
        conversation.append({"role": "user", "content": user_message})
        
        is_done = False
        iteration = 0
        
        while not is_done and iteration < self.max_react_iterations:
            iteration += 1
            
            try:
                # Get LLM response (Thought + Action)
                llm_response = await self._call_llm_for_react(
                    system_prompt, 
                    conversation
                )
                
                logger.info(f"ReAct iteration {iteration}: {llm_response[:100]}...")
                
                # Parse Thought and Action from response
                thought, action_str = self._parse_react_response(llm_response)
                
                if thought:
                    result.reasoning_steps.append(f"Thought: {thought}")
                
                if not action_str:
                    result.reasoning_steps.append("Không tìm thấy Action trong response")
                    break
                
                result.reasoning_steps.append(f"Action: {action_str}")
                
                # Add assistant response to conversation
                conversation.append({"role": "assistant", "content": llm_response})
                
                # Execute the action
                tool_result = await self._execute_action(action_str)
                
                # Check if done
                if tool_result.data == "__DONE__":
                    is_done = True
                    result.reasoning_steps.append(f"Done: {tool_result.error}")  # error contains summary
                    continue
                
                # Collect data from tool result
                self._collect_tool_data(tool_result, collected_data)
                
                # Create observation
                observation = tool_result.to_observation()
                result.reasoning_steps.append(f"Observation: {observation[:200]}...")
                
                # Add observation to conversation
                conversation.append({"role": "user", "content": f"Observation: {observation}"})
                
            except Exception as e:
                logger.error(f"ReAct iteration {iteration} error: {e}")
                result.reasoning_steps.append(f"Error: {str(e)}")
                break
        
        # Compile results
        result.nodes = collected_data["nodes"]
        result.paths = collected_data["paths"]
        result.community_summaries = [c.get("summary", "") for c in collected_data["communities"]]
        
        # Set confidence based on collected data
        if is_done:
            result.confidence = 0.9
        elif result.nodes or result.paths:
            result.confidence = 0.7
        else:
            result.confidence = 0.4
        
        result.synthesized_context = result.to_context_string()
        
        return result
    
    async def _call_llm_for_react(
        self, 
        system_prompt: str, 
        conversation: List[Dict[str, str]]
    ) -> str:
        """Call LLM with ReAct prompt."""
        if not self.llm_port:
            raise ValueError("LLM port not configured")
        
        # Format messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        
        try:
            # Use configured react_model if available
            generate_kwargs = {
                "messages": messages,
                "temperature": 0.3,  # Low temperature for more deterministic reasoning
                "max_tokens": 500
            }
            if self.react_model:
                generate_kwargs["model"] = self.react_model
                
            response = await self.llm_port.generate(**generate_kwargs)
            return response.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    def _parse_react_response(self, response: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse Thought and Action from LLM response.
        
        Expected format:
        Thought: <reasoning>
        Action: <tool_call>
        """
        thought = None
        action = None
        
        # Extract Thought
        thought_match = re.search(r"Thought:\s*(.+?)(?=Action:|$)", response, re.DOTALL | re.IGNORECASE)
        if thought_match:
            thought = thought_match.group(1).strip()
        
        # Extract Action
        action_match = re.search(r"Action:\s*(.+?)(?=Observation:|Thought:|$)", response, re.DOTALL | re.IGNORECASE)
        if action_match:
            action = action_match.group(1).strip()
            # Clean up action - take only the first line if multiple
            action = action.split("\n")[0].strip()
        
        return thought, action
    
    async def _execute_action(self, action_str: str) -> ToolResult:
        """
        Execute an action string (tool call).
        
        Parses action like: search_articles("đăng ký, học phần")
        """
        # Parse tool name and arguments
        match = re.match(r"(\w+)\s*\(([^)]*)\)", action_str)
        
        if not match:
            return ToolResult(
                success=False, 
                error=f"Invalid action format: {action_str}"
            )
        
        tool_name = match.group(1)
        args_str = match.group(2).strip()
        
        # Get tool function
        tool_func = self._tools_map.get(tool_name)
        if not tool_func:
            return ToolResult(
                success=False, 
                error=f"Unknown tool: {tool_name}"
            )
        
        # Parse arguments
        try:
            args = self._parse_tool_args(args_str)
            result = await tool_func(*args)
            return result
        except Exception as e:
            logger.error(f"Tool execution error: {e}")
            return ToolResult(success=False, error=str(e))
    
    def _parse_tool_args(self, args_str: str) -> List[Any]:
        """Parse tool arguments from string."""
        if not args_str:
            return []
        
        args = []
        # Handle quoted strings and numbers
        # Split by comma, but handle quoted strings properly
        parts = []
        current = ""
        in_quotes = False
        quote_char = None
        
        for char in args_str:
            if char in '"\'':
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
                else:
                    current += char
            elif char == ',' and not in_quotes:
                parts.append(current.strip())
                current = ""
            else:
                current += char
        
        if current:
            parts.append(current.strip())
        
        for part in parts:
            part = part.strip().strip('"\'')
            # Try to convert to number
            try:
                if '.' in part:
                    args.append(float(part))
                else:
                    args.append(int(part))
            except ValueError:
                args.append(part)
        
        return args
    
    def _collect_tool_data(self, tool_result: ToolResult, collected_data: Dict):
        """Collect data from tool result into accumulated data."""
        if not tool_result.success or not tool_result.data:
            return
        
        data = tool_result.data
        
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    item_type = item.get("type") or item.get("node_type", "")
                    if item_type == "Article":
                        collected_data["articles"].append(item)
                        collected_data["nodes"].append(item)
                    elif item_type == "Community":
                        collected_data["communities"].append(item)
                    elif item_type in ("path", "Path"):
                        collected_data["paths"].append(item)
                    else:
                        collected_data["nodes"].append(item)
        elif isinstance(data, dict):
            if data.get("type") == "Article":
                collected_data["articles"].append(data)
                collected_data["nodes"].append(data)
            elif data.get("type") == "Community":
                collected_data["communities"].append(data)
    
    # ========== TOOL IMPLEMENTATIONS ==========
    
    async def _tool_search_articles(self, keywords: str) -> ToolResult:
        """Tool: Search articles by keywords, automatically including amendment info."""
        try:
            keyword_list = [k.strip() for k in keywords.split(",")]
            
            # Use search_with_amendments to get both original and amended content
            if hasattr(self.graph_adapter, 'search_with_amendments'):
                results = await self.graph_adapter.search_with_amendments(keyword_list, limit=5)
            else:
                # Fallback to regular search if method not available
                results = await self.graph_adapter.search_articles_by_keyword(keyword_list, limit=5)
            
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_check_amendments(self, article_title: str) -> ToolResult:
        """Tool: Check if an article has been amended and get the latest version."""
        try:
            if hasattr(self.graph_adapter, 'get_latest_version_of_article'):
                result = await self.graph_adapter.get_latest_version_of_article(article_title)
                if result["is_amended"]:
                    return ToolResult(
                        success=True, 
                        data={
                            "message": f"Điều '{article_title}' đã được sửa đổi",
                            "amendment_description": result["amendment_description"],
                            "amending_article": result["amending_article"],
                            "original_article": result["original_article"]
                        }
                    )
                else:
                    return ToolResult(
                        success=True, 
                        data={"message": f"Điều '{article_title}' chưa có sửa đổi", "original_article": result["original_article"]}
                    )
            else:
                return ToolResult(success=False, error="Amendment check not available")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_search_entities(self, keywords: str) -> ToolResult:
        """Tool: Search entities by keywords."""
        try:
            keyword_list = [k.strip() for k in keywords.split(",")]
            results = await self.graph_adapter.search_entities_by_keyword(keyword_list, limit=10)
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_get_article(self, article_number: int) -> ToolResult:
        """Tool: Get article by number."""
        try:
            result = await self.graph_adapter.get_article_with_entities(article_number)
            if result:
                return ToolResult(success=True, data=result)
            else:
                return ToolResult(success=False, error=f"Không tìm thấy Điều {article_number}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_get_related_articles(self, entity_name: str) -> ToolResult:
        """Tool: Get articles related to an entity."""
        try:
            results = await self.graph_adapter.get_related_articles_by_entity(entity_name, limit=5)
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_get_communities(self) -> ToolResult:
        """Tool: Get all communities."""
        try:
            results = await self.graph_adapter.get_all_communities()
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_find_article_path(self, start_article: int, end_article: int) -> ToolResult:
        """Tool: Find path between two articles."""
        try:
            results = await self.graph_adapter.find_article_path(
                start_article, end_article, max_depth=5
            )
            return ToolResult(success=True, data=results)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_find_prerequisites(self, course_code: str) -> ToolResult:
        """Tool: Find prerequisites for a course."""
        try:
            paths = await self.graph_adapter.find_prerequisites_chain(course_code, max_depth=3)
            if paths:
                result = self._convert_paths_to_dict(paths)
                return ToolResult(success=True, data=result)
            else:
                return ToolResult(success=False, error=f"Không tìm thấy môn tiên quyết cho {course_code}")
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_find_dependents(self, course_code: str) -> ToolResult:
        """Tool: Find courses that depend on this course."""
        try:
            # First find the node
            nodes = await self._find_nodes_by_codes([course_code])
            if not nodes:
                return ToolResult(success=False, error=f"Không tìm thấy môn {course_code}")
            
            # Then explore dependents
            paths = await self._explore_dependents(
                nodes[0]["id"], 
                [], 
                max_depth=self.max_hop_depth, 
                visited=set()
            )
            return ToolResult(success=True, data=paths)
        except Exception as e:
            return ToolResult(success=False, error=str(e))
    
    async def _tool_scan_relationships(self, rel_types: str) -> ToolResult:
        """
        Tool: Scan all node pairs with specific relationship types.
        
        Args:
            rel_types: Comma-separated relationship types (e.g., "YEU_CAU, QUY_DINH_DIEU_KIEN")
            
        Returns:
            ToolResult with list of relationship pairs
        """
        try:
            # Parse relationship types
            type_list = [t.strip().upper() for t in rel_types.split(",")]
            
            # Call adapter to get pairs
            pairs = await self.graph_adapter.get_pairs_by_relationship_type(type_list, limit=50)
            
            # Format results for better readability
            formatted_results = []
            for pair in pairs:
                formatted_results.append({
                    "source": f"[{pair['source']['type']}] {pair['source']['name']}",
                    "relationship": pair['relationship'],
                    "target": f"[{pair['target']['type']}] {pair['target']['name']}"
                })
            
            return ToolResult(success=True, data=formatted_results)
        except Exception as e:
            logger.error(f"Error scanning relationships: {e}")
            return ToolResult(success=False, error=str(e))
    
    async def _tool_done(self, summary: str = "") -> ToolResult:
        """Tool: Signal that reasoning is complete."""
        return ToolResult(success=True, data="__DONE__", error=summary)
    
    async def reason(
        self,
        query: str,
        query_type: GraphQueryType,
        context: Optional[Dict[str, Any]] = None
    ) -> GraphReasoningResult:
        """
        Main entry point for graph reasoning.
        
        Args:
            query: User query
            query_type: Type of graph query (local/global/multi_hop)
            context: Additional context (extracted entities, filters, etc.)
            
        Returns:
            GraphReasoningResult with synthesized context
        """
        context = context or {}
        
        logger.info(f"Graph reasoning: type={query_type.value}, query={query[:50]}...")
        
        if query_type == GraphQueryType.LOCAL:
            return await self._local_reasoning(query, context)
        elif query_type == GraphQueryType.GLOBAL:
            return await self._global_reasoning(query, context)
        elif query_type == GraphQueryType.MULTI_HOP:
            return await self._multi_hop_reasoning(query, context)
        else:
            logger.warning(f"Unknown query type: {query_type}, falling back to local")
            return await self._local_reasoning(query, context)
    
    # ========== LOCAL REASONING (1-HOP) ==========
    
    async def _local_reasoning(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> GraphReasoningResult:
        """
        Local reasoning: Pattern-based 1-hop queries.
        
        Handles:
        - Prerequisites: "Môn X cần môn gì?"
        - Department courses: "Các môn của khoa Y?"
        - Program requirements: "Môn bắt buộc của chương trình Z?"
        """
        result = GraphReasoningResult(
            query_type=GraphQueryType.LOCAL,
            query=query
        )
        
        # Extract entity from query
        entity = self._extract_entity_from_query(query)
        query_pattern = self._detect_query_pattern(query)
        
        logger.info(f"Local reasoning: entity={entity}, pattern={query_pattern}")
        
        try:
            # === ENHANCED: Nếu không tìm thấy mã môn, thử tìm bằng tên ===
            if not entity and query_pattern in ["prerequisite", "general"]:
                logger.info("No course code found, trying full-text search on MON_HOC")
                try:
                    # NodeCategory imported at top of file
                    mon_hoc_nodes = await self.graph_adapter.search_nodes(
                        query, 
                        categories=[NodeCategory.MON_HOC], 
                        limit=3
                    )
                    
                    if mon_hoc_nodes:
                        # Lấy mã môn từ node đầu tiên
                        first_node = mon_hoc_nodes[0]
                        entity = first_node.properties.get("ma_mon") or first_node.properties.get("code")
                        
                        logger.info(f"Found course via full-text search: {entity}")
                        result.reasoning_steps.append(
                            f"Tìm thấy môn học '{first_node.properties.get('ten_mon', entity)}' qua tìm kiếm tên"
                        )
                        
                        # Nếu pattern chưa rõ, set thành prerequisite
                        if query_pattern == "general":
                            query_pattern = "prerequisite"
                except Exception as e:
                    logger.warning(f"Full-text search on MON_HOC failed: {e}")
            
            if query_pattern == "prerequisite" and entity:
                # Pattern 1: Find prerequisites
                paths = await self.graph_adapter.find_prerequisites_chain(entity, max_depth=3)
                
                if paths:
                    result.paths = self._convert_paths_to_dict(paths)
                    result.nodes = self._extract_nodes_from_paths(paths)
                    result.reasoning_steps = [
                        f"Tìm chuỗi môn tiên quyết cho {entity}",
                        f"Phát hiện {len(paths)} đường đi tiên quyết",
                    ]
                    result.confidence = 0.9
                else:
                    result.reasoning_steps = [f"Không tìm thấy môn tiên quyết cho {entity}"]
                    result.confidence = 0.5
            
            elif query_pattern == "department" and entity:
                # Pattern 2: Find courses in department
                nodes = await self._find_courses_by_department(entity)
                
                if nodes:
                    result.nodes = nodes
                    result.reasoning_steps = [
                        f"Tìm các môn học thuộc khoa {entity}",
                        f"Tìm thấy {len(nodes)} môn học"
                    ]
                    result.confidence = 0.85
                else:
                    result.reasoning_steps = [f"Không tìm thấy môn học cho khoa {entity}"]
                    result.confidence = 0.4
            
            elif query_pattern == "program" and entity:
                # Pattern 3: Find required courses in program
                nodes = await self._find_courses_by_program(entity)
                
                if nodes:
                    result.nodes = nodes
                    result.reasoning_steps = [
                        f"Tìm các môn học của chương trình {entity}",
                        f"Tìm thấy {len(nodes)} môn học"
                    ]
                    result.confidence = 0.85
                else:
                    result.reasoning_steps = [f"Không tìm thấy môn học cho chương trình {entity}"]
                    result.confidence = 0.4
            
            elif query_pattern == "article" and entity:
                # Pattern 4: Find specific Article by number (CatRAG)
                article_num = self._extract_article_number(query)
                if article_num:
                    article = await self.graph_adapter.get_article_with_entities(article_num)
                    if article:
                        result.nodes = [article]
                        result.reasoning_steps = [
                            f"Tìm Điều {article_num} trong quy chế",
                            f"Tìm thấy với {len(article.get('entities', []))} thực thể liên quan"
                        ]
                        result.confidence = 0.95
                    else:
                        result.reasoning_steps = [f"Không tìm thấy Điều {article_num}"]
                        result.confidence = 0.3
            
            elif query_pattern == "entity_related" and entity:
                # Pattern 5: Find articles related to an entity (CatRAG)
                articles = await self.graph_adapter.get_related_articles_by_entity(entity, limit=5)
                if articles:
                    result.nodes = articles
                    result.reasoning_steps = [
                        f"Tìm các điều khoản liên quan đến '{entity}'",
                        f"Tìm thấy {len(articles)} điều khoản"
                    ]
                    result.confidence = 0.85
                else:
                    result.reasoning_steps = [f"Không tìm thấy điều khoản liên quan đến '{entity}'"]
                    result.confidence = 0.4
            
            else:
                # === ENHANCED FALLBACK: Tìm kiếm toàn diện ===
                keywords = self._extract_keywords_from_query(query)
                
                # If no keywords extracted and LLM available, try ReAct
                if not keywords and self.llm_port:
                    logger.info("No keywords extracted, trying ReAct")
                    return await self._react_loop(query, context)
                
                # 1. Search Articles by keywords (Quy chế)
                articles = await self.graph_adapter.search_articles_by_keyword(keywords, limit=5)
                
                # 2. Search Entities by keywords
                entities = await self.graph_adapter.search_entities_by_keyword(keywords, limit=5)
                
                # 3. ENHANCED: Search MON_HOC nodes specifically
                mon_hoc_nodes = []
                try:
                    # NodeCategory imported at top of file
                    mon_hoc_search = await self.graph_adapter.search_nodes(
                        query, 
                        categories=[NodeCategory.MON_HOC], 
                        limit=10
                    )
                    
                    # Convert GraphNode to dict format
                    for node in mon_hoc_search:
                        mon_hoc_nodes.append({
                            "id": node.id,
                            "name": node.properties.get("ten_mon", "Unknown"),
                            "ma_mon": node.properties.get("ma_mon", ""),
                            "so_tin_chi": node.properties.get("so_tin_chi", 0),
                            "type": "MON_HOC",
                            "node_type": "Course"
                        })
                    
                    logger.info(f"Found {len(mon_hoc_nodes)} MON_HOC nodes via full-text search")
                except Exception as e:
                    logger.warning(f"MON_HOC search failed: {e}")
                
                # Combine all results
                all_nodes = articles + entities + mon_hoc_nodes
                
                if all_nodes:
                    result.nodes = all_nodes
                    result.reasoning_steps = [
                        f"Tìm kiếm với từ khóa: {', '.join(keywords)}",
                        f"Tìm thấy {len(articles)} điều khoản, {len(entities)} thực thể, {len(mon_hoc_nodes)} môn học"
                    ]
                    result.confidence = 0.7
                elif self.llm_port:
                    # No results from keyword search, try ReAct
                    logger.info("No keyword results, falling back to ReAct")
                    return await self._react_loop(query, context)
                else:
                    result.reasoning_steps = ["Không tìm thấy kết quả phù hợp"]
                    result.confidence = 0.3
        
        except Exception as e:
            logger.error(f"Local reasoning error: {e}")
            result.reasoning_steps = [f"Lỗi: {str(e)}"]
            result.confidence = 0.0
        
        # Synthesize context
        result.synthesized_context = result.to_context_string()
        
        return result
    
    # ========== GLOBAL REASONING (COMMUNITY-BASED) ==========
    
    async def _global_reasoning(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> GraphReasoningResult:
        """
        Global reasoning: Community-based summaries for overview/comparative questions.
        
        Uses pre-computed Community Reports from build_communities.py.
        
        Handles:
        - "So sánh khác biệt giữa chương trình CNTT và KHMT"
        - "Tóm tắt các quy định về học vụ"
        - "Tổng quan về các môn học AI"
        """
        result = GraphReasoningResult(
            query_type=GraphQueryType.GLOBAL,
            query=query
        )
        
        try:
            # Step 1: Find relevant communities based on query
            communities = await self._find_relevant_communities(query)
            
            if communities:
                result.community_summaries = [
                    c.get("full_summary", c.get("label", ""))
                    for c in communities
                ]
                
                result.reasoning_steps = [
                    f"Phân tích query: '{query[:50]}...'",
                    f"Tìm thấy {len(communities)} community liên quan",
                    "Sử dụng Community Reports để tổng hợp thông tin toàn cục"
                ]
                
                # Extract nodes from communities
                for comm in communities:
                    articles = comm.get("articles", [])
                    for art in articles[:5]:  # Limit per community
                        result.nodes.append({
                            "id": art.get("id"),
                            "name": art.get("title", art.get("id")),
                            "type": "Article",
                            "community": comm.get("label")
                        })
                
                result.confidence = 0.8
            else:
                # Fallback: Use entity-based aggregation
                result.reasoning_steps = [
                    "Không tìm thấy Community Reports phù hợp",
                    "Thực hiện aggregation dựa trên entities"
                ]
                
                aggregated = await self._aggregate_by_entities(query)
                if aggregated:
                    result.nodes = aggregated
                    result.confidence = 0.6
                else:
                    result.confidence = 0.3
        
        except Exception as e:
            logger.error(f"Global reasoning error: {e}")
            result.reasoning_steps = [f"Lỗi: {str(e)}"]
            result.confidence = 0.0
        
        # Synthesize context
        result.synthesized_context = result.to_context_string()
        
        return result
    
    async def _find_relevant_communities(self, query: str) -> List[Dict[str, Any]]:
        """
        Find communities relevant to the query using semantic matching.
        
        Queries the Community nodes created by build_communities.py.
        """
        try:
            # Query Community nodes with full_summary
            cypher = """
            MATCH (c:Community)
            WHERE c.full_summary IS NOT NULL AND c.full_summary <> ''
            RETURN c.id as id, c.label as label, c.full_summary as full_summary, 
                   c.size as size, c.key_entities as key_entities
            ORDER BY c.size DESC
            LIMIT $limit
            """
            
            results = await self.graph_adapter.execute_cypher(cypher, {"limit": 10})
            
            if not results:
                return []
            
            # Score communities by keyword matching with query
            query_lower = query.lower()
            scored_communities = []
            
            for comm in results:
                score = 0
                summary = (comm.get("full_summary") or "").lower()
                label = (comm.get("label") or "").lower()
                
                # Simple keyword matching (could be enhanced with embeddings)
                query_words = set(query_lower.split())
                summary_words = set(summary.split())
                label_words = set(label.split())
                
                # Score based on word overlap
                summary_overlap = len(query_words & summary_words)
                label_overlap = len(query_words & label_words)
                
                score = summary_overlap * 2 + label_overlap * 3
                
                if score > 0:
                    scored_communities.append((score, comm))
            
            # Sort by score and return top communities
            scored_communities.sort(key=lambda x: x[0], reverse=True)
            
            return [comm for score, comm in scored_communities[:self.max_communities]]
        
        except Exception as e:
            logger.error(f"Error finding communities: {e}")
            return []
    
    # ========== MULTI-HOP REASONING (DYNAMIC) ==========
    
    async def _multi_hop_reasoning(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> GraphReasoningResult:
        """
        Multi-hop reasoning: Dynamic path exploration with ReAct framework.
        
        When LLM is available:
        - Uses ReAct (Reasoning + Acting) loop
        - LLM decides which tools to call based on reasoning
        - Supports complex queries that don't match static patterns
        
        When LLM is not available:
        - Falls back to pattern-based multi-hop exploration
        
        Handles:
        - "Nếu tôi rớt IT001 thì tôi sẽ bị trễ những môn đồ án nào?"
        - "Môn này ảnh hưởng thế nào đến khả năng tốt nghiệp?"
        - "Chuỗi môn học từ cơ sở đến chuyên ngành AI?"
        """
        # Try ReAct first if LLM is available
        if self.llm_port:
            logger.info("Using ReAct framework for multi-hop reasoning")
            return await self._react_loop(query, context)
        
        # Fallback to pattern-based reasoning
        logger.info("LLM not available, using pattern-based multi-hop reasoning")
        return await self._pattern_based_multi_hop(query, context)
    
    async def _pattern_based_multi_hop(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> GraphReasoningResult:
        """
        Pattern-based multi-hop reasoning (fallback when LLM not available).
        
        Uses hardcoded patterns to extract entities and explore graph.
        """
        result = GraphReasoningResult(
            query_type=GraphQueryType.MULTI_HOP,
            query=query
        )
        
        try:
            # Step 1: Extract starting entities and target concepts
            start_entities, target_concepts = self._parse_multi_hop_query(query)
            
            result.reasoning_steps.append(
                f"Phân tích query: start={start_entities}, target={target_concepts}"
            )
            
            if not start_entities:
                result.reasoning_steps.append("Không xác định được điểm bắt đầu")
                result.confidence = 0.3
                result.synthesized_context = result.to_context_string()
                return result
            
            # Step 2: Find starting node(s)
            start_nodes = await self._find_nodes_by_codes(start_entities)
            
            if not start_nodes:
                result.reasoning_steps.append(f"Không tìm thấy node cho {start_entities}")
                result.confidence = 0.3
                result.synthesized_context = result.to_context_string()
                return result
            
            result.reasoning_steps.append(f"Tìm thấy {len(start_nodes)} node khởi đầu")
            
            # Step 3: Dynamic multi-hop exploration
            all_paths = []
            visited_nodes = set()
            
            for start_node in start_nodes:
                # Explore in REVERSE direction: What courses DEPEND on this course?
                dependent_paths = await self._explore_dependents(
                    start_node["id"],
                    target_concepts,
                    max_depth=self.max_hop_depth,
                    visited=visited_nodes
                )
                all_paths.extend(dependent_paths)
                
                result.reasoning_steps.append(
                    f"Từ {start_node.get('ma_mon', 'N/A')}: tìm thấy {len(dependent_paths)} đường đi"
                )
            
            # Step 4: Filter paths based on target concepts
            filtered_paths = self._filter_paths_by_target(all_paths, target_concepts)
            
            result.paths = filtered_paths
            result.reasoning_steps.append(
                f"Sau khi lọc theo target '{target_concepts}': {len(filtered_paths)} đường đi"
            )
            
            # Step 5: Extract all affected nodes
            result.nodes = self._extract_unique_nodes_from_paths(filtered_paths)
            
            # Confidence based on results
            if filtered_paths:
                result.confidence = 0.85
            elif all_paths:
                result.paths = all_paths[:10]  # Return unfiltered if no target match
                result.confidence = 0.7
            else:
                result.confidence = 0.4
        
        except Exception as e:
            logger.error(f"Multi-hop reasoning error: {e}")
            result.reasoning_steps.append(f"Lỗi: {str(e)}")
            result.confidence = 0.0
        
        # Synthesize context
        result.synthesized_context = result.to_context_string()
        
        return result
    
    async def _explore_dependents(
        self,
        node_id: str,
        target_concepts: List[str],
        max_depth: int,
        visited: set
    ) -> List[Dict[str, Any]]:
        """
        Explore nodes that DEPEND on the given node (reverse direction).
        
        This answers: "What courses will be affected if I fail this course?"
        
        Uses dynamic Cypher instead of hardcoded patterns.
        """
        # Dynamic query: Find all courses that have this as prerequisite (any depth)
        cypher = f"""
        MATCH path = (dependent:MON_HOC)-[:DIEU_KIEN_TIEN_QUYET*1..{max_depth}]->(source)
        WHERE elementId(source) = $node_id
        WITH path, 
             [node IN nodes(path) | node.ma_mon] as course_codes,
             [node IN nodes(path) | node.ten_mon] as course_names,
             length(path) as depth
        RETURN course_codes, course_names, depth
        ORDER BY depth
        """
        
        try:
            results = await self.graph_adapter.execute_cypher(cypher, {"node_id": node_id})
            
            paths = []
            for r in results:
                codes = r.get("course_codes", [])
                names = r.get("course_names", [])
                depth = r.get("depth", 0)
                
                # Skip if already visited
                path_key = "->".join(codes)
                if path_key in visited:
                    continue
                visited.add(path_key)
                
                paths.append({
                    "node_codes": codes,
                    "node_names": names,
                    "depth": depth,
                    "direction": "dependent"  # These courses depend on the source
                })
            
            return paths
        
        except Exception as e:
            logger.error(f"Error exploring dependents: {e}")
            return []
    
    def _parse_multi_hop_query(self, query: str) -> Tuple[List[str], List[str]]:
        """
        Parse multi-hop query to extract start entities and target concepts.
        
        Examples:
        - "Nếu tôi rớt IT001 thì tôi sẽ bị trễ những môn đồ án nào?"
          → start: ["IT001"], target: ["đồ án"]
        - "Môn IT003 ảnh hưởng đến những môn nào năm cuối?"
          → start: ["IT003"], target: ["năm cuối"]
        """
        query_lower = query.lower()
        
        # Extract course codes (IT001, SE104, etc.)
        course_pattern = r'\b([A-Z]{2}\d{3})\b'
        start_entities = re.findall(course_pattern, query.upper())
        
        # Extract target concepts
        target_patterns = [
            (r'đồ\s*án', 'đồ án'),
            (r'chuyên\s*ngành', 'chuyên ngành'),
            (r'tốt\s*nghiệp', 'tốt nghiệp'),
            (r'năm\s*cuối', 'năm cuối'),
            (r'năm\s*(\d+)', 'năm'),
            (r'học\s*kỳ\s*(\d+)', 'học kỳ'),
        ]
        
        target_concepts = []
        for pattern, concept in target_patterns:
            if re.search(pattern, query_lower):
                target_concepts.append(concept)
        
        return start_entities, target_concepts
    
    def _filter_paths_by_target(
        self,
        paths: List[Dict[str, Any]],
        target_concepts: List[str]
    ) -> List[Dict[str, Any]]:
        """Filter paths that lead to target concepts."""
        if not target_concepts:
            return paths
        
        filtered = []
        for path in paths:
            names = path.get("node_names", [])
            names_lower = [n.lower() if n else "" for n in names]
            
            # Check if any target concept appears in path
            for target in target_concepts:
                target_lower = target.lower()
                for name in names_lower:
                    if target_lower in name:
                        filtered.append(path)
                        break
                else:
                    continue
                break
        
        return filtered
    
    # ========== HELPER METHODS ==========
    
    def _extract_entity_from_query(self, query: str) -> Optional[str]:
        """Extract main entity (course code, department, etc.) from query."""
        # Course code pattern
        course_match = re.search(r'\b([A-Z]{2}\d{3})\b', query.upper())
        if course_match:
            return course_match.group(1)
        
        # Department patterns
        dept_patterns = {
            "cntt": "CNTT",
            "công nghệ thông tin": "CNTT",
            "khmt": "KHMT",
            "khoa học máy tính": "KHMT",
            "httt": "HTTT",
            "hệ thống thông tin": "HTTT",
            "mmt": "MMT",
            "mạng máy tính": "MMT"
        }
        
        query_lower = query.lower()
        for pattern, code in dept_patterns.items():
            if pattern in query_lower:
                return code
        
        return None
    
    def _detect_query_pattern(self, query: str) -> str:
        """Detect the type of query pattern."""
        query_lower = query.lower()
        
        # Article-specific patterns (CatRAG - "Điều 14", "Điều số 10")
        article_patterns = [
            r"điều\s*\d+", r"điều\s*số\s*\d+", r"article\s*\d+"
        ]
        if any(re.search(p, query_lower) for p in article_patterns):
            return "article"
        
        # Entity-related patterns (CatRAG - "liên quan đến X", "về X")
        entity_related_patterns = [
            "liên quan đến", "liên quan tới", "quy định về", 
            "các điều khoản về", "nói về", "đề cập đến"
        ]
        if any(p in query_lower for p in entity_related_patterns):
            return "entity_related"
        
        # Prerequisite patterns
        prereq_patterns = ["tiên quyết", "học trước", "cần học", "môn trước"]
        if any(p in query_lower for p in prereq_patterns):
            return "prerequisite"
        
        # Department patterns
        dept_patterns = ["thuộc khoa", "của khoa", "trong khoa"]
        if any(p in query_lower for p in dept_patterns):
            return "department"
        
        # Program patterns
        program_patterns = ["chương trình", "ngành học", "ctđt"]
        if any(p in query_lower for p in program_patterns):
            return "program"
        
        return "general"
    
    def _extract_article_number(self, query: str) -> Optional[int]:
        """Extract article number from query like 'Điều 14' or 'Điều số 10'."""
        patterns = [
            r"điều\s*(\d+)",
            r"điều\s*số\s*(\d+)",
            r"article\s*(\d+)"
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return int(match.group(1))
        return None
    
    def _extract_keywords_from_query(self, query: str) -> List[str]:
        """Extract meaningful keywords from query for CatRAG search."""
        # Remove common Vietnamese stop words
        stop_words = {
            "là", "gì", "như", "thế", "nào", "và", "hoặc", "hay", 
            "của", "cho", "với", "để", "từ", "đến", "trong", "ngoài",
            "các", "những", "một", "hai", "ba", "có", "không", "được",
            "phải", "cần", "nên", "hỏi", "về", "tôi", "bạn", "em", "anh"
        }
        
        # Split and filter
        words = query.lower().split()
        keywords = [w for w in words if len(w) > 2 and w not in stop_words]
        
        # Also extract compound terms (Vietnamese compound words)
        compound_terms = []
        compound_patterns = [
            r"học phần", r"môn học", r"sinh viên", r"tín chỉ", r"điểm trung bình",
            r"đăng ký", r"xét tốt nghiệp", r"tốt nghiệp", r"buộc thôi học",
            r"cảnh báo học tập", r"kết quả", r"chương trình đào tạo",
            r"tiên quyết", r"học trước", r"môn cơ sở", r"môn chuyên ngành",
            # ENHANCED: Thêm tên môn học phổ biến
            r"nhập môn lập trình", r"lập trình hướng đối tượng",
            r"cấu trúc dữ liệu", r"giải thuật", r"cơ sở dữ liệu",
            r"mạng máy tính", r"hệ điều hành", r"công nghệ phần mềm",
            r"trí tuệ nhân tạo", r"machine learning", r"deep learning",
            # ENHANCED: Thêm ngoại ngữ và quy định
            r"tiếng anh", r"tiếng nhật", r"tiếng pháp", r"ngoại ngữ",
            r"quy định", r"quy chế", r"điều kiện", r"yêu cầu",
            r"miễn học", r"xét miễn", r"chứng chỉ", r"toeic",
            r"đào tạo ngoại ngữ", r"giảng dạy tiếng anh"
        ]
        for pattern in compound_patterns:
            if re.search(pattern, query.lower()):
                compound_terms.append(pattern.replace(r" ", " "))
        
        # ENHANCED: If compound terms found, prioritize them over single keywords
        # This prevents searching with fragmented keywords like "tiếng", "anh" separately
        if compound_terms:
            # Return compound terms first, then add remaining single keywords
            remaining_keywords = [k for k in keywords if not any(k in ct for ct in compound_terms)]
            return list(set(compound_terms + remaining_keywords[:5]))[:15]
        
        # ENHANCED: Extract course names without codes
        # Pattern: "Môn [Tên]" or just "[Tên môn học]"
        course_name_pattern = r"môn\s+([A-ZĐÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ][a-zđàáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵ\s]+)"
        course_matches = re.findall(course_name_pattern, query, re.IGNORECASE)
        for match in course_matches:
            compound_terms.append(match.strip())
        
        return list(set(keywords + compound_terms))[:15]  # Max 15 keywords
    
    async def _find_courses_by_department(self, dept_code: str) -> List[Dict[str, Any]]:
        """Find courses belonging to a department."""
        cypher = """
        MATCH (course:MON_HOC)-[:THUOC_KHOA]->(dept:KHOA {ma_khoa: $dept_code})
        RETURN course.ma_mon as code, course.ten_mon as name, 
               course.so_tin_chi as credits, 'MON_HOC' as type
        ORDER BY course.ma_mon
        LIMIT 20
        """
        
        try:
            results = await self.graph_adapter.execute_cypher(cypher, {"dept_code": dept_code})
            return results
        except Exception as e:
            logger.error(f"Error finding courses by department: {e}")
            return []
    
    async def _find_courses_by_program(self, program_code: str) -> List[Dict[str, Any]]:
        """Find courses in a program."""
        cypher = """
        MATCH (course:MON_HOC)-[r:THUOC_CHUONG_TRINH]->(prog:CHUONG_TRINH_DAO_TAO)
        WHERE prog.ma_chuong_trinh CONTAINS $program_code
        RETURN course.ma_mon as code, course.ten_mon as name,
               course.so_tin_chi as credits, r.loai_mon as course_type,
               'MON_HOC' as type
        ORDER BY r.hoc_ky_khuyen_nghi, course.ma_mon
        LIMIT 30
        """
        
        try:
            results = await self.graph_adapter.execute_cypher(cypher, {"program_code": program_code})
            return results
        except Exception as e:
            logger.error(f"Error finding courses by program: {e}")
            return []
    
    async def _find_nodes_by_codes(self, codes: List[str]) -> List[Dict[str, Any]]:
        """Find nodes by course codes."""
        cypher = """
        MATCH (n:MON_HOC)
        WHERE n.ma_mon IN $codes
        RETURN elementId(n) as id, n.ma_mon as ma_mon, n.ten_mon as ten_mon,
               n.so_tin_chi as so_tin_chi
        """
        
        try:
            results = await self.graph_adapter.execute_cypher(cypher, {"codes": codes})
            return results
        except Exception as e:
            logger.error(f"Error finding nodes by codes: {e}")
            return []
    
    async def _aggregate_by_entities(self, query: str) -> List[Dict[str, Any]]:
        """Aggregate nodes by common entities when no community matches."""
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        if not keywords:
            return []
        
        # Search for nodes matching keywords
        cypher = """
        CALL db.index.fulltext.queryNodes('mon_hoc_fulltext', $query)
        YIELD node, score
        RETURN elementId(node) as id, node.ma_mon as code, node.ten_mon as name,
               score, 'MON_HOC' as type
        ORDER BY score DESC
        LIMIT 10
        """
        
        try:
            results = await self.graph_adapter.execute_cypher(
                cypher, 
                {"query": " OR ".join(keywords)}
            )
            return results
        except Exception as e:
            logger.error(f"Error aggregating by entities: {e}")
            return []
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Vietnamese stop words
        stop_words = {
            "là", "của", "và", "có", "trong", "với", "để", "về", "tại", "từ",
            "này", "đó", "những", "các", "một", "như", "được", "sẽ", "đã",
            "đang", "thì", "nếu", "khi", "mà", "hay", "hoặc", "gì", "nào",
            "so", "sánh", "khác", "biệt", "giữa", "tôi", "bạn"
        }
        
        words = query.lower().replace(",", " ").replace(".", " ").split()
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords[:5]  # Limit to 5 keywords
    
    def _convert_paths_to_dict(self, paths) -> List[Dict[str, Any]]:
        """Convert GraphPath objects to dictionaries."""
        result = []
        for path in paths:
            node_names = []
            node_codes = []
            
            for node in path.nodes:
                props = node.properties
                name = props.get("ten_mon") or props.get("name", "Unknown")
                code = props.get("ma_mon") or props.get("code", "")
                node_names.append(name)
                node_codes.append(code)
            
            result.append({
                "node_names": node_names,
                "node_codes": node_codes,
                "length": path.length
            })
        
        return result
    
    def _extract_nodes_from_paths(self, paths) -> List[Dict[str, Any]]:
        """Extract unique nodes from paths."""
        seen = set()
        nodes = []
        
        for path in paths:
            for node in path.nodes:
                node_id = node.id
                if node_id not in seen:
                    seen.add(node_id)
                    nodes.append(self._node_to_dict(node))
        
        return nodes
    
    def _extract_unique_nodes_from_paths(self, paths: List[Dict]) -> List[Dict[str, Any]]:
        """Extract unique nodes from path dictionaries."""
        seen = set()
        nodes = []
        
        for path in paths:
            codes = path.get("node_codes", [])
            names = path.get("node_names", [])
            
            for code, name in zip(codes, names):
                if code and code not in seen:
                    seen.add(code)
                    nodes.append({
                        "ma_mon": code,
                        "name": name,
                        "type": "MON_HOC"
                    })
        
        return nodes
    
    def _node_to_dict(self, node) -> Dict[str, Any]:
        """Convert GraphNode to dictionary."""
        props = node.properties.copy()
        props["id"] = node.id
        props["type"] = node.category.value if hasattr(node.category, 'value') else str(node.category)
        
        # Ensure name field exists
        if "name" not in props:
            props["name"] = props.get("ten_mon") or props.get("ma_mon", "Unknown")
        
        return props
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information."""
        return {
            "name": "GraphReasoningAgent",
            "version": "1.0.0",
            "capabilities": [
                "local_reasoning",
                "global_reasoning", 
                "multi_hop_reasoning"
            ],
            "max_hop_depth": self.max_hop_depth,
            "max_communities": self.max_communities
        }

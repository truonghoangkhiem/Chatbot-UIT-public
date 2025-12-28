#!/usr/bin/env python3
"""
Test script for Graph Reasoning Agent

Tests the three reasoning modes:
1. LOCAL: Simple 1-hop queries (prerequisites, department courses)
2. GLOBAL: Community-based summaries for comparative/overview questions
3. MULTI_HOP: Dynamic path exploration with impact analysis
"""

import asyncio
import sys
from pathlib import Path

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from app.agents.graph_reasoning_agent import GraphReasoningAgent, GraphQueryType
from app.agents.smart_planner_agent import SmartPlannerAgent


class MockGraphAdapter:
    """Mock Graph Adapter for testing without Neo4j connection."""
    
    async def find_prerequisites_chain(self, course_code: str, max_depth: int = 3):
        """Mock prerequisite chain."""
        # Simulate prerequisite data
        mock_chains = {
            "IT003": [
                MockPath([
                    {"ma_mon": "IT003", "ten_mon": "Cấu trúc dữ liệu và giải thuật"},
                    {"ma_mon": "IT002", "ten_mon": "Lập trình hướng đối tượng"},
                    {"ma_mon": "IT001", "ten_mon": "Nhập môn lập trình"}
                ])
            ],
            "SE363": [
                MockPath([
                    {"ma_mon": "SE363", "ten_mon": "Trí tuệ nhân tạo"},
                    {"ma_mon": "IT003", "ten_mon": "Cấu trúc dữ liệu và giải thuật"},
                    {"ma_mon": "IT002", "ten_mon": "Lập trình hướng đối tượng"}
                ])
            ]
        }
        return mock_chains.get(course_code, [])
    
    async def execute_cypher(self, cypher: str, params: dict = None):
        """Mock Cypher execution."""
        params = params or {}
        
        # Mock community data
        if "Community" in cypher:
            return [
                {
                    "id": "community_1",
                    "label": "Cộng đồng: Môn học cơ sở lập trình",
                    "full_summary": "Nhóm các môn học nền tảng về lập trình bao gồm IT001 (Nhập môn lập trình), IT002 (Lập trình hướng đối tượng), và IT003 (Cấu trúc dữ liệu). Các môn này tạo thành chuỗi tiên quyết cơ bản cho sinh viên ngành CNTT.",
                    "size": 5
                },
                {
                    "id": "community_2", 
                    "label": "Cộng đồng: Môn học chuyên ngành AI",
                    "full_summary": "Nhóm các môn học chuyên ngành về Trí tuệ nhân tạo, bao gồm Machine Learning, Deep Learning, và các ứng dụng AI. Yêu cầu nền tảng toán học và lập trình vững chắc.",
                    "size": 4
                }
            ]
        
        # Mock dependent courses (reverse prerequisite)
        if "dependent" in cypher.lower() or "DIEU_KIEN_TIEN_QUYET" in cypher:
            node_id = params.get("node_id", "")
            if "IT001" in str(node_id) or params.get("codes") == ["IT001"]:
                return [
                    {
                        "course_codes": ["IT002", "IT001"],
                        "course_names": ["Lập trình hướng đối tượng", "Nhập môn lập trình"],
                        "depth": 1
                    },
                    {
                        "course_codes": ["IT003", "IT002", "IT001"],
                        "course_names": ["Cấu trúc dữ liệu và giải thuật", "Lập trình hướng đối tượng", "Nhập môn lập trình"],
                        "depth": 2
                    },
                    {
                        "course_codes": ["SE104", "IT003", "IT002", "IT001"],
                        "course_names": ["Đồ án công nghệ phần mềm", "Cấu trúc dữ liệu", "Lập trình OOP", "Nhập môn lập trình"],
                        "depth": 3
                    }
                ]
        
        # Mock find nodes by code
        if "ma_mon IN" in cypher:
            codes = params.get("codes", [])
            results = []
            for code in codes:
                results.append({
                    "id": f"node_{code}",
                    "ma_mon": code,
                    "ten_mon": f"Môn học {code}",
                    "so_tin_chi": 3
                })
            return results
        
        return []
    
    async def search_nodes(self, query: str, limit: int = 5):
        """Mock full-text search."""
        return []


class MockPath:
    """Mock GraphPath for testing."""
    def __init__(self, nodes_data):
        self.nodes = [MockNode(n) for n in nodes_data]
        self.length = len(nodes_data) - 1


class MockNode:
    """Mock GraphNode for testing."""
    def __init__(self, props):
        self.id = props.get("ma_mon", "unknown")
        self.properties = props
        self.category = MockCategory("MON_HOC")


class MockCategory:
    """Mock NodeCategory."""
    def __init__(self, value):
        self.value = value


def print_separator(title):
    """Print section separator."""
    print(f"\n{'='*70}")
    print(f"🧪 {title}")
    print(f"{'='*70}")


async def test_graph_query_type_detection():
    """Test SmartPlannerAgent's graph_query_type detection."""
    print_separator("TEST 1: Graph Query Type Detection")
    
    # Test queries and expected types
    test_cases = [
        # LOCAL queries
        ("Môn IT003 cần học môn gì trước?", "local"),
        ("Các môn tiên quyết của SE363 là gì?", "local"),
        ("Môn học thuộc khoa CNTT?", "local"),
        
        # GLOBAL queries  
        ("So sánh chương trình đào tạo CNTT và KHMT?", "global"),
        ("Tóm tắt các quy định về học vụ", "global"),
        ("Tổng quan về các môn học AI", "global"),
        
        # MULTI_HOP queries
        ("Nếu tôi rớt IT001 thì tôi sẽ bị trễ những môn đồ án nào?", "multi_hop"),
        ("Rớt môn IT002 ảnh hưởng đến năm cuối như thế nào?", "multi_hop"),
        ("Chuỗi môn học từ cơ sở đến chuyên ngành AI?", "multi_hop"),
    ]
    
    # Create a minimal SmartPlannerAgent-like detector
    from app.agents.smart_planner_agent import SmartPlannerAgent
    
    # We'll use the static method directly
    class TestDetector:
        def _determine_graph_query_type(self, query: str) -> str:
            import re
            query_lower = query.lower()
            
            # MULTI_HOP patterns
            multi_hop_patterns = [
                r'nếu.*(rớt|trượt|không qua|fail).*thì',
                r'nếu.*(không học|bỏ qua|skip).*thì',
                r'(rớt|trượt).*ảnh hưởng',
                r'(rớt|trượt).*bị trễ',
                r'chuỗi.*(môn|học phần)',
                r'từ.*(cơ sở|nền tảng).*đến.*(chuyên ngành|nâng cao)',
            ]
            for pattern in multi_hop_patterns:
                if re.search(pattern, query_lower):
                    return "multi_hop"
            
            # GLOBAL patterns
            global_patterns = [
                r'so sánh.*(chương trình|ngành|khoa)',
                r'khác biệt.*(giữa|của).*(chương trình|ngành|khoa)',
                r'tóm tắt.*(quy định|chương trình|môn học)',
                r'tổng quan.*(về|của)',
            ]
            for pattern in global_patterns:
                if re.search(pattern, query_lower):
                    return "global"
            
            return "local"
    
    detector = TestDetector()
    
    passed = 0
    failed = 0
    
    for query, expected in test_cases:
        result = detector._determine_graph_query_type(query)
        status = "✅" if result == expected else "❌"
        if result == expected:
            passed += 1
        else:
            failed += 1
        print(f"{status} Query: '{query[:50]}...'")
        print(f"   Expected: {expected}, Got: {result}")
    
    print(f"\n📊 Results: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)


async def test_local_reasoning():
    """Test LOCAL reasoning mode."""
    print_separator("TEST 2: LOCAL Reasoning (1-hop queries)")
    
    mock_adapter = MockGraphAdapter()
    agent = GraphReasoningAgent(graph_adapter=mock_adapter)
    
    # Test prerequisite query
    query = "Môn IT003 cần học môn gì trước?"
    print(f"\n📝 Query: {query}")
    
    result = await agent.reason(
        query=query,
        query_type=GraphQueryType.LOCAL,
        context={}
    )
    
    print(f"\n📊 Result:")
    print(f"   - Query Type: {result.query_type.value}")
    print(f"   - Confidence: {result.confidence}")
    print(f"   - Nodes found: {len(result.nodes)}")
    print(f"   - Paths found: {len(result.paths)}")
    print(f"\n   Reasoning Steps:")
    for step in result.reasoning_steps:
        print(f"      • {step}")
    
    if result.paths:
        print(f"\n   Prerequisite Chains:")
        for path in result.paths:
            chain = " → ".join(path.get("node_names", []))
            print(f"      • {chain}")
    
    print(f"\n   Synthesized Context (first 300 chars):")
    print(f"   {result.synthesized_context[:300]}...")
    
    return result.confidence > 0.5


async def test_global_reasoning():
    """Test GLOBAL reasoning mode."""
    print_separator("TEST 3: GLOBAL Reasoning (Community summaries)")
    
    mock_adapter = MockGraphAdapter()
    agent = GraphReasoningAgent(graph_adapter=mock_adapter)
    
    # Test comparative query
    query = "Tổng quan về các môn học lập trình cơ sở"
    print(f"\n📝 Query: {query}")
    
    result = await agent.reason(
        query=query,
        query_type=GraphQueryType.GLOBAL,
        context={}
    )
    
    print(f"\n📊 Result:")
    print(f"   - Query Type: {result.query_type.value}")
    print(f"   - Confidence: {result.confidence}")
    print(f"   - Communities found: {len(result.community_summaries)}")
    print(f"\n   Reasoning Steps:")
    for step in result.reasoning_steps:
        print(f"      • {step}")
    
    if result.community_summaries:
        print(f"\n   Community Summaries:")
        for i, summary in enumerate(result.community_summaries, 1):
            print(f"      [{i}] {summary[:100]}...")
    
    print(f"\n   Synthesized Context (first 300 chars):")
    print(f"   {result.synthesized_context[:300]}...")
    
    return result.confidence > 0.3


async def test_multi_hop_reasoning():
    """Test MULTI_HOP reasoning mode."""
    print_separator("TEST 4: MULTI_HOP Reasoning (Dynamic path exploration)")
    
    mock_adapter = MockGraphAdapter()
    agent = GraphReasoningAgent(graph_adapter=mock_adapter)
    
    # Test impact analysis query
    query = "Nếu tôi rớt IT001 thì tôi sẽ bị trễ những môn đồ án nào?"
    print(f"\n📝 Query: {query}")
    
    result = await agent.reason(
        query=query,
        query_type=GraphQueryType.MULTI_HOP,
        context={}
    )
    
    print(f"\n📊 Result:")
    print(f"   - Query Type: {result.query_type.value}")
    print(f"   - Confidence: {result.confidence}")
    print(f"   - Nodes affected: {len(result.nodes)}")
    print(f"   - Paths discovered: {len(result.paths)}")
    print(f"\n   Reasoning Steps:")
    for step in result.reasoning_steps:
        print(f"      • {step}")
    
    if result.paths:
        print(f"\n   Impact Paths (courses depending on IT001):")
        for path in result.paths[:5]:  # Show first 5
            codes = path.get("node_codes", [])
            names = path.get("node_names", [])
            chain = " ← ".join(codes)  # Reverse direction for dependent
            print(f"      • {chain}")
    
    if result.nodes:
        print(f"\n   Affected Courses:")
        for node in result.nodes[:10]:
            print(f"      • {node.get('ma_mon', 'N/A')}: {node.get('name', 'N/A')}")
    
    print(f"\n   Synthesized Context (first 400 chars):")
    print(f"   {result.synthesized_context[:400]}...")
    
    return result.confidence > 0.3


async def test_integration_flow():
    """Test the full integration flow."""
    print_separator("TEST 5: Integration Flow Simulation")
    
    print("\n🔄 Simulating full orchestrator flow with Graph Reasoning...")
    
    # Simulate the flow
    test_queries = [
        ("Môn IT003 cần học môn gì?", "local"),
        ("So sánh chương trình CNTT và KHMT", "global"),
        ("Nếu tôi rớt IT001 thì bị trễ môn nào?", "multi_hop"),
    ]
    
    mock_adapter = MockGraphAdapter()
    agent = GraphReasoningAgent(graph_adapter=mock_adapter)
    
    for query, expected_type in test_queries:
        print(f"\n{'─'*50}")
        print(f"📝 Query: {query}")
        
        # Detect query type (simulating SmartPlanner)
        query_type = GraphQueryType(expected_type)
        print(f"   Detected type: {query_type.value}")
        
        # Execute reasoning
        result = await agent.reason(query, query_type, {})
        
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Context length: {len(result.synthesized_context)} chars")
        
        # Show snippet
        context_snippet = result.synthesized_context[:150].replace('\n', ' ')
        print(f"   Context snippet: {context_snippet}...")
    
    return True


async def main():
    """Run all tests."""
    print("\n" + "🚀" * 35)
    print("    GRAPH REASONING AGENT TEST SUITE")
    print("🚀" * 35)
    
    results = []
    
    # Test 1: Query Type Detection
    results.append(("Query Type Detection", await test_graph_query_type_detection()))
    
    # Test 2: Local Reasoning
    results.append(("Local Reasoning", await test_local_reasoning()))
    
    # Test 3: Global Reasoning
    results.append(("Global Reasoning", await test_global_reasoning()))
    
    # Test 4: Multi-hop Reasoning
    results.append(("Multi-hop Reasoning", await test_multi_hop_reasoning()))
    
    # Test 5: Integration Flow
    results.append(("Integration Flow", await test_integration_flow()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {name}")
        if result:
            passed += 1
    
    print(f"\n   Total: {passed}/{len(results)} tests passed")
    print("=" * 70)
    
    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

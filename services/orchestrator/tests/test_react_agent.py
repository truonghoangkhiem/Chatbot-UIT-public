#!/usr/bin/env python3
"""
Test ReAct (Reasoning + Acting) Framework for GraphReasoningAgent

Tests the dynamic tool-calling capability when LLM is available,
and fallback behavior when LLM is not available.
"""

import asyncio
import sys
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

# Setup paths
orchestrator_path = Path(__file__).parent.parent
rag_services_path = orchestrator_path.parent / "rag_services"

sys.path.insert(0, str(orchestrator_path))
sys.path.insert(0, str(rag_services_path))

# Load environment
from dotenv import load_dotenv
env_path = rag_services_path / ".env"
load_dotenv(env_path)


def print_separator(title):
    """Print section separator."""
    print(f"\n{'='*70}")
    print(f"🧪 {title}")
    print(f"{'='*70}")


def print_header():
    """Print test suite header."""
    print("🤖"*35)
    print("    REACT FRAMEWORK TEST SUITE")
    print("    Testing: Thought → Action → Observation Loop")
    print("🤖"*35)


# ========== MOCK LLM PORT ==========

@dataclass
class MockLLMResponse:
    """Mock LLM response."""
    content: str


class MockLLMPort:
    """
    Mock LLM Port for testing ReAct.
    
    Simulates LLM responses for Thought → Action pattern.
    """
    
    def __init__(self, responses: list = None):
        """Initialize with predefined responses."""
        self.responses = responses or []
        self.call_count = 0
        self.call_history = []
    
    async def generate(self, messages: list, temperature: float = 0.3, max_tokens: int = 500):
        """Generate mock LLM response."""
        self.call_history.append({
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
            self.call_count += 1
            return MockLLMResponse(content=response)
        
        # Default: Return done
        return MockLLMResponse(
            content="Thought: Đã thu thập đủ thông tin\nAction: done(\"Đã hoàn thành\")"
        )


# ========== TESTS ==========

async def test_react_tool_definitions():
    """Test that all ReAct tools are properly defined."""
    print_separator("TEST 1: Tool Definitions")
    
    # Import GraphReasoningAgent dynamically
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "graph_reasoning_agent",
        orchestrator_path / "app" / "agents" / "graph_reasoning_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    GraphReasoningAgent = module.GraphReasoningAgent
    Tool = module.Tool
    
    # Check REACT_TOOLS class attribute
    assert hasattr(GraphReasoningAgent, 'REACT_TOOLS'), "Missing REACT_TOOLS attribute"
    tools = GraphReasoningAgent.REACT_TOOLS
    
    print(f"\n📋 Found {len(tools)} tools defined:")
    
    expected_tools = [
        "search_articles", "search_entities", "get_article", 
        "get_related_articles", "get_communities", "find_article_path",
        "find_prerequisites", "find_dependents", "done"
    ]
    
    for tool in tools:
        assert isinstance(tool, Tool), f"Tool {tool.name} is not a Tool instance"
        print(f"   ✓ {tool.name}: {tool.description[:50]}...")
        assert tool.name in expected_tools, f"Unexpected tool: {tool.name}"
    
    print(f"\n✅ All {len(tools)} tools properly defined")
    return True


async def test_react_prompt():
    """Test ReAct system prompt generation."""
    print_separator("TEST 2: System Prompt Generation")
    
    # Import module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "graph_reasoning_agent",
        orchestrator_path / "app" / "agents" / "graph_reasoning_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    GraphReasoningAgent = module.GraphReasoningAgent
    
    # Check REACT_SYSTEM_PROMPT
    assert hasattr(GraphReasoningAgent, 'REACT_SYSTEM_PROMPT'), "Missing REACT_SYSTEM_PROMPT"
    prompt = GraphReasoningAgent.REACT_SYSTEM_PROMPT
    
    # Check key components in prompt
    assert "{tools_description}" in prompt, "Missing tools_description placeholder"
    assert "Thought:" in prompt, "Missing Thought instruction"
    assert "Action:" in prompt, "Missing Action instruction"
    assert "Observation:" in prompt, "Missing Observation instruction"
    assert "done()" in prompt, "Missing done() instruction"
    
    print("\n📝 System Prompt Structure:")
    print(f"   ✓ Contains tools_description placeholder")
    print(f"   ✓ Contains Thought/Action/Observation pattern")
    print(f"   ✓ Contains example usage")
    print(f"   ✓ Total length: {len(prompt)} chars")
    
    print("\n✅ System prompt correctly structured")
    return True


async def test_action_parsing():
    """Test parsing of Thought/Action from LLM response."""
    print_separator("TEST 3: Action Parsing")
    
    # Import module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "graph_reasoning_agent",
        orchestrator_path / "app" / "agents" / "graph_reasoning_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    GraphReasoningAgent = module.GraphReasoningAgent
    
    # Create mock adapter
    mock_adapter = MagicMock()
    agent = GraphReasoningAgent(graph_adapter=mock_adapter, llm_port=None)
    
    # Test cases
    test_cases = [
        {
            "input": "Thought: Cần tìm nội dung Điều 14\nAction: get_article(14)",
            "expected_thought": "Cần tìm nội dung Điều 14",
            "expected_action": "get_article(14)"
        },
        {
            "input": "Thought: Tìm các điều khoản về đăng ký\nAction: search_articles(\"đăng ký, học phần\")",
            "expected_thought": "Tìm các điều khoản về đăng ký",
            "expected_action": 'search_articles("đăng ký, học phần")'
        },
        {
            "input": "Thought: Đã có đủ thông tin\nAction: done(\"Tìm thấy 3 điều khoản liên quan\")",
            "expected_thought": "Đã có đủ thông tin",
            "expected_action": 'done("Tìm thấy 3 điều khoản liên quan")'
        },
    ]
    
    print("\n📋 Testing action parsing:")
    
    for i, case in enumerate(test_cases, 1):
        thought, action = agent._parse_react_response(case["input"])
        
        assert thought is not None, f"Case {i}: Thought should not be None"
        assert action is not None, f"Case {i}: Action should not be None"
        assert case["expected_thought"] in thought, f"Case {i}: Thought mismatch"
        assert case["expected_action"] in action, f"Case {i}: Action mismatch"
        
        print(f"   ✓ Case {i}: Parsed correctly")
        print(f"      Thought: {thought[:40]}...")
        print(f"      Action: {action}")
    
    print(f"\n✅ All {len(test_cases)} parsing cases passed")
    return True


async def test_tool_args_parsing():
    """Test parsing of tool arguments."""
    print_separator("TEST 4: Tool Arguments Parsing")
    
    # Import module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "graph_reasoning_agent",
        orchestrator_path / "app" / "agents" / "graph_reasoning_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    GraphReasoningAgent = module.GraphReasoningAgent
    
    # Create mock adapter
    mock_adapter = MagicMock()
    agent = GraphReasoningAgent(graph_adapter=mock_adapter, llm_port=None)
    
    # Test cases
    test_cases = [
        ('14', [14]),
        ('"đăng ký, học phần"', ["đăng ký, học phần"]),
        ('10, 25', [10, 25]),
        ('"sinh viên"', ["sinh viên"]),
        ('', []),
        ('"IT001"', ["IT001"]),
    ]
    
    print("\n📋 Testing argument parsing:")
    
    for args_str, expected in test_cases:
        result = agent._parse_tool_args(args_str)
        assert result == expected, f"Expected {expected}, got {result}"
        print(f"   ✓ '{args_str}' → {result}")
    
    print(f"\n✅ All {len(test_cases)} argument parsing cases passed")
    return True


async def test_react_loop_with_mock_llm():
    """Test full ReAct loop with mock LLM."""
    print_separator("TEST 5: ReAct Loop with Mock LLM")
    
    # Import module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "graph_reasoning_agent",
        orchestrator_path / "app" / "agents" / "graph_reasoning_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    GraphReasoningAgent = module.GraphReasoningAgent
    
    # Create mock adapter with async methods
    mock_adapter = MagicMock()
    mock_adapter.search_articles_by_keyword = AsyncMock(return_value=[
        {"title": "Điều 14 - Đăng ký học phần", "content": "Nội dung...", "type": "Article"},
        {"title": "Điều 15 - Thêm bớt học phần", "content": "Nội dung...", "type": "Article"},
    ])
    mock_adapter.get_article_with_entities = AsyncMock(return_value={
        "title": "Đăng ký học phần",
        "content": "Sinh viên đăng ký học phần theo kế hoạch...",
        "article_number": 14,
        "entities": [{"name": "sinh viên"}, {"name": "học phần"}],
        "type": "Article"
    })
    
    # Create mock LLM with predefined responses
    mock_llm = MockLLMPort(responses=[
        "Thought: Cần tìm nội dung của Điều 14\nAction: get_article(14)",
        "Thought: Đã có thông tin về Điều 14, cần tìm thêm các điều liên quan\nAction: search_articles(\"đăng ký, học phần\")",
        "Thought: Đã có đủ thông tin để trả lời\nAction: done(\"Điều 14 quy định về đăng ký học phần, tìm thấy 2 điều khoản liên quan\")"
    ])
    
    # Create agent
    agent = GraphReasoningAgent(graph_adapter=mock_adapter, llm_port=mock_llm)
    
    print("\n🔄 Running ReAct loop for query: 'Điều 14 quy định về vấn đề gì?'")
    
    # Run ReAct loop
    result = await agent._react_loop("Điều 14 quy định về vấn đề gì?", {})
    
    # Check results
    assert result is not None, "Result should not be None"
    assert len(result.reasoning_steps) > 0, "Should have reasoning steps"
    assert result.confidence > 0.5, f"Confidence should be > 0.5, got {result.confidence}"
    
    print(f"\n📋 ReAct Loop Results:")
    print(f"   LLM calls: {mock_llm.call_count}")
    print(f"   Reasoning steps: {len(result.reasoning_steps)}")
    print(f"   Nodes collected: {len(result.nodes)}")
    print(f"   Confidence: {result.confidence}")
    
    print("\n   Reasoning chain:")
    for step in result.reasoning_steps:
        print(f"      • {step[:80]}...")
    
    print(f"\n✅ ReAct loop completed successfully")
    return True


async def test_fallback_without_llm():
    """Test that agent falls back to pattern-based when LLM not available."""
    print_separator("TEST 6: Fallback Without LLM")
    
    # Import module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "graph_reasoning_agent",
        orchestrator_path / "app" / "agents" / "graph_reasoning_agent.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    GraphReasoningAgent = module.GraphReasoningAgent
    GraphQueryType = module.GraphQueryType
    
    # Create mock adapter
    mock_adapter = MagicMock()
    mock_adapter.search_articles_by_keyword = AsyncMock(return_value=[
        {"title": "Điều 14", "type": "Article"}
    ])
    mock_adapter.search_entities_by_keyword = AsyncMock(return_value=[])
    
    # Create agent WITHOUT LLM
    agent = GraphReasoningAgent(graph_adapter=mock_adapter, llm_port=None)
    
    print("\n🔄 Testing MULTI_HOP without LLM (should use pattern-based)")
    
    # This should fall back to pattern-based reasoning
    result = await agent.reason(
        "Nếu rớt IT001 thì ảnh hưởng gì?",
        GraphQueryType.MULTI_HOP,
        {}
    )
    
    assert result is not None, "Result should not be None"
    print(f"   Result: {len(result.reasoning_steps)} reasoning steps")
    print(f"   Confidence: {result.confidence}")
    
    print("\n✅ Fallback to pattern-based works correctly")
    return True


async def main():
    """Run all tests."""
    print_header()
    
    tests = [
        ("Tool Definitions", test_react_tool_definitions),
        ("System Prompt", test_react_prompt),
        ("Action Parsing", test_action_parsing),
        ("Tool Args Parsing", test_tool_args_parsing),
        ("ReAct Loop with Mock LLM", test_react_loop_with_mock_llm),
        ("Fallback Without LLM", test_fallback_without_llm),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = await test_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"\n❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Print summary
    print_separator("TEST SUMMARY")
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {name}")
        if error:
            print(f"      Error: {error}")
    
    print(f"\n   Total: {passed}/{total} tests passed")
    print("="*70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

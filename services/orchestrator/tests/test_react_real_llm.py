#!/usr/bin/env python3
"""
Test ReAct Framework với LLM thật (OpenRouter) - Sử dụng cấu hình hệ thống

Chạy: cd services/orchestrator && python tests/test_react_real_llm.py
"""

import asyncio
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime

# Setup paths BEFORE any imports
orchestrator_path = Path(__file__).parent.parent.resolve()
rag_services_path = orchestrator_path.parent / "rag_services"

# IMPORTANT: Insert orchestrator FIRST so its 'app' package takes priority
# over rag_services/app
sys.path.insert(0, str(rag_services_path))  # Second priority
sys.path.insert(0, str(orchestrator_path))  # First priority (will be checked first)

# Load environment
from dotenv import load_dotenv
env_path = orchestrator_path / ".env"
if not env_path.exists():
    env_path = rag_services_path / ".env"
load_dotenv(env_path)

# Import from packages now that path is set
from app.adapters.openrouter_adapter import OpenRouterAdapter
from app.core.domain import (
    AgentRequest, ConversationContext, ConversationMessage, 
    ConversationRole, MessageType
)
from app.agents.graph_reasoning_agent import GraphReasoningAgent, GraphQueryType
from adapters.graph.neo4j_adapter import Neo4jGraphAdapter


def print_separator(title):
    """Print section separator."""
    print(f"\n{'='*70}")
    print(f"🧪 {title}")
    print(f"{'='*70}")


def print_header():
    """Print test suite header."""
    print("🤖"*35)
    print("    REACT + REAL LLM TEST SUITE")
    print("    Using: OpenRouterAdapter from Orchestrator")
    print("🤖"*35)


# ========== LLM PORT WRAPPER ==========

class LLMPortWrapper:
    """
    Wrapper để adapt OpenRouterAdapter interface cho GraphReasoningAgent.
    
    GraphReasoningAgent gọi: llm_port.generate(messages=..., temperature=..., max_tokens=...)
    OpenRouterAdapter cần: generate_response(AgentRequest)
    """
    
    def __init__(self, openrouter_adapter):
        """Initialize wrapper with OpenRouterAdapter."""
        self.adapter = openrouter_adapter
        self.call_count = 0
    
    async def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.3, 
        max_tokens: int = 500
    ):
        """
        Generate response - adapts to GraphReasoningAgent's expected interface.
        """
        self.call_count += 1
        
        # Extract system prompt and conversation messages
        system_prompt = None
        conv_messages = []
        user_prompt = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                user_prompt = content  # Last user message becomes prompt
                conv_messages.append(ConversationMessage(
                    role=ConversationRole.USER,
                    content=content,
                    timestamp=datetime.now(),
                    message_type=MessageType.TEXT
                ))
            elif role == "assistant":
                conv_messages.append(ConversationMessage(
                    role=ConversationRole.ASSISTANT,
                    content=content,
                    timestamp=datetime.now(),
                    message_type=MessageType.TEXT
                ))
        
        # Create context
        context = ConversationContext(
            session_id="react-test",
            messages=conv_messages[:-1] if conv_messages else [],
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        # Create request
        request = AgentRequest(
            prompt=user_prompt,
            context=context,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False
        )
        
        print(f"   📤 LLM call #{self.call_count}")
        
        # Call adapter
        response = await self.adapter.generate_response(request)
        
        print(f"      Tokens: {response.tokens_used}, Time: {response.processing_time:.2f}s")
        
        return response  # AgentResponse has .content field


# ========== TEST FUNCTIONS ==========

async def test_llm_connection():
    """Test LLM connection using OpenRouterAdapter."""
    print_separator("TEST 1: LLM Connection (OpenRouterAdapter)")
    
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found (OPENROUTER_API_KEY or OPENAI_API_KEY)")
        return False
    
    print(f"   API Key: {api_key[:20]}...")
    
    adapter = OpenRouterAdapter(
        api_key=api_key,
        default_model="google/gemini-2.0-flash-001",
        timeout=60
    )
    
    # Simple test
    request = AgentRequest(
        prompt="Trả lời ngắn gọn: 1 + 1 = ?",
        temperature=0.1,
        max_tokens=50
    )
    
    response = await adapter.generate_response(request)
    
    print(f"\n📝 Response: {response.content}")
    print(f"   Model: {response.model_used}")
    print(f"   Tokens: {response.tokens_used}")
    
    await adapter.close()
    
    assert response.content, "Response should not be empty"
    print("\n✅ LLM connection OK")
    return True


async def test_react_with_system_llm():
    """Test ReAct loop với LLM từ hệ thống và Neo4j thật."""
    print_separator("TEST 2: ReAct Loop với System LLM")
    
    # Get API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False
    
    print("\n📡 Initializing components...")
    
    # Neo4j
    neo4j_adapter = Neo4jGraphAdapter(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    
    healthy = await neo4j_adapter.health_check()
    if not healthy:
        print("❌ Neo4j not available, skipping test")
        return False
    print("   ✓ Neo4j connected")
    
    # LLM
    openrouter = OpenRouterAdapter(
        api_key=api_key,
        default_model="google/gemini-2.0-flash-001",
        timeout=60
    )
    llm_port = LLMPortWrapper(openrouter)
    print("   ✓ OpenRouterAdapter connected")
    
    # Create agent
    agent = GraphReasoningAgent(
        graph_adapter=neo4j_adapter,
        llm_port=llm_port
    )
    print("   ✓ GraphReasoningAgent created with ReAct")
    
    # Test queries
    test_queries = [
        {
            "query": "Điều 14 của quy chế đào tạo quy định về vấn đề gì?",
            "type": GraphQueryType.MULTI_HOP,
        },
    ]
    
    results = []
    
    for i, test in enumerate(test_queries, 1):
        print(f"\n{'─'*60}")
        print(f"📝 Query {i}: {test['query']}")
        print(f"{'─'*60}")
        
        result = await agent.reason(
            query=test["query"],
            query_type=test["type"],
            context={}
        )
        
        print(f"\n📊 Results:")
        print(f"   Confidence: {result.confidence:.2f}")
        print(f"   Nodes found: {len(result.nodes)}")
        print(f"   Paths found: {len(result.paths)}")
        print(f"   LLM calls: {llm_port.call_count}")
        
        print(f"\n🔄 Reasoning Chain ({len(result.reasoning_steps)} steps):")
        for j, step in enumerate(result.reasoning_steps, 1):
            step_display = step[:100] + "..." if len(step) > 100 else step
            print(f"   {j}. {step_display}")
        
        if result.nodes:
            print(f"\n📄 Sample Nodes:")
            for node in result.nodes[:3]:
                name = node.get("title") or node.get("name") or "Unknown"
                print(f"      • {name[:60]}...")
        
        results.append({
            "query": test["query"],
            "success": result.confidence > 0.3,
            "confidence": result.confidence,
            "steps": len(result.reasoning_steps),
            "nodes": len(result.nodes)
        })
    
    # Cleanup
    await openrouter.close()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"   {status} {r['query'][:40]}... (conf={r['confidence']:.2f}, nodes={r['nodes']})")
    
    all_passed = all(r["success"] for r in results)
    print(f"\n   Total LLM calls: {llm_port.call_count}")
    print(f"   Result: {'✅ ALL PASSED' if all_passed else '⚠️ SOME FAILED'}")
    
    return all_passed


async def test_react_complex_query():
    """Test ReAct với câu hỏi phức tạp."""
    print_separator("TEST 3: Complex Query")
    
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found")
        return False
    
    # Initialize
    neo4j_adapter = Neo4jGraphAdapter(
        uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        username=os.getenv("NEO4J_USER", "neo4j"),
        password=os.getenv("NEO4J_PASSWORD", "password")
    )
    
    healthy = await neo4j_adapter.health_check()
    if not healthy:
        print("❌ Neo4j not available")
        return False
    
    openrouter = OpenRouterAdapter(
        api_key=api_key,
        default_model="google/gemini-2.0-flash-001",
        timeout=60
    )
    llm_port = LLMPortWrapper(openrouter)
    agent = GraphReasoningAgent(graph_adapter=neo4j_adapter, llm_port=llm_port)
    
    # Complex query
    query = "Sinh viên bị buộc thôi học trong những trường hợp nào?"
    
    print(f"\n📝 Query: {query}")
    
    result = await agent.reason(
        query=query,
        query_type=GraphQueryType.MULTI_HOP,
        context={}
    )
    
    print(f"\n🔄 Reasoning ({len(result.reasoning_steps)} steps):")
    for i, step in enumerate(result.reasoning_steps, 1):
        print(f"   {i}. {step[:80]}...")
    
    print(f"\n   Confidence: {result.confidence:.2f}")
    print(f"   Nodes: {len(result.nodes)}")
    print(f"   LLM calls: {llm_port.call_count}")
    
    # Cleanup
    await openrouter.close()
    
    print("\n✅ Complex query test completed")
    return result.confidence > 0.3


async def main():
    """Run all tests."""
    print_header()
    
    # Check API key
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ No API key found in environment")
        print("   Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        return False
    
    tests = [
        ("LLM Connection", test_llm_connection),
        ("ReAct with System LLM", test_react_with_system_llm),
        ("Complex Query Test", test_react_complex_query),
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
    
    # Final summary
    print(f"\n{'='*70}")
    print(f"🧪 FINAL SUMMARY")
    print(f"{'='*70}")
    
    passed = 0
    for name, success, error in results:
        if success:
            print(f"   ✅ PASSED: {name}")
            passed += 1
        else:
            print(f"   ❌ FAILED: {name}")
            if error:
                print(f"      Error: {error[:50]}...")
    
    print(f"\n   Total: {passed}/{len(results)} tests passed")
    print(f"{'='*70}")
    
    return passed == len(results)


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)

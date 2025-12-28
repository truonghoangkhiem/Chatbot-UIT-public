#!/usr/bin/env python3
"""
Comprehensive test for optimized 3-agent pipeline.
Tests: SmartPlannerAgent, AnswerAgent, ResponseFormatterAgent
"""

import asyncio
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file first
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)

# Add the app directory to Python path
sys.path.append('/home/kien/Kien/study/SE363.Q11/Chatbot-UIT/services/orchestrator')

from app.adapters.openrouter_adapter import OpenRouterAdapter
from app.core.domain import AgentRequest, ConversationContext


async def test_smart_planner_agent():
    """
    Test SmartPlannerAgent - Merged Planner + Query Rewriter.
    Handles: intent classification, complexity scoring, query rewriting in single LLM call.
    """
    print("🧠 Testing Smart Planner Agent (merged planner + query rewriter)...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ API key not found!")
        return False
    
    try:
        adapter = OpenRouterAdapter(api_key=api_key, timeout=None)
        
        context = ConversationContext(
            session_id="test-smart-planner",
            messages=[],
            system_prompt="""Bạn là SmartPlannerAgent cho hệ thống Chatbot-UIT.
Nhiệm vụ của bạn là:
1. Phân loại intent của câu hỏi (academic, administrative, general)
2. Đánh giá độ phức tạp (1-10)
3. Viết lại câu hỏi tối ưu cho RAG search
4. Xác định cần KG và/hoặc Vector search

Trả lời bằng JSON format:
{
    "intent": "academic|administrative|general",
    "complexity": 1-10,
    "rewritten_queries": ["query1", "query2"],
    "use_knowledge_graph": true/false,
    "use_vector_search": true/false,
    "reasoning": "brief explanation"
}"""
        )
        
        request = AgentRequest(
            prompt="Tôi muốn tìm hiểu về học phí và cách đăng ký học phần tại UIT",
            context=context,
            model="mistralai/mistral-7b-instruct:free",
            temperature=0.1,
            max_tokens=500
        )
        
        response = await adapter.generate_response(request)
        
        print(f"✅ Smart Planner Agent Response:")
        print(f"Model: {response.model_used}")
        print(f"Content: {response.content[:500]}...")
        print(f"Tokens: {response.tokens_used}")
        
        await adapter.close()
        return True
        
    except Exception as e:
        print(f"❌ Smart Planner Agent Error: {e}")
        return False


async def test_answer_agent():
    """
    Test AnswerAgent - Core reasoning agent.
    Generates answers based on RAG context.
    """
    print("\n💬 Testing Answer Agent...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ API key not found!")
        return False
    
    try:
        adapter = OpenRouterAdapter(api_key=api_key, timeout=None)
        
        # Simulate RAG context
        rag_context = [
            "Học phí tại Đại học Công nghệ Thông tin (UIT) năm 2024: Sinh viên công lập: 756.000 VNĐ/tín chỉ.",
            "Thời gian đăng ký học phần: Thường vào tháng 7-8 cho học kỳ 1, tháng 12-1 cho học kỳ 2.",
            "Sinh viên cần đăng nhập vào hệ thống Portal để đăng ký học phần trực tuyến."
        ]
        
        context = ConversationContext(
            session_id="test-answer",
            messages=[],
            system_prompt=f"""Bạn là Answer Agent chuyên trả lời câu hỏi dựa trên context được cung cấp.
Sử dụng thông tin sau để trả lời chính xác và hữu ích:

CONTEXT:
{chr(10).join(rag_context)}

Trả lời bằng tiếng Việt, tự nhiên và thân thiện."""
        )
        
        request = AgentRequest(
            prompt="Học phí UIT bao nhiêu tiền? Và làm thế nào để đăng ký học phần?",
            context=context,
            model="google/gemma-2-9b-it:free",
            temperature=0.7,
            max_tokens=400
        )
        
        response = await adapter.generate_response(request)
        
        print(f"✅ Answer Agent Response:")
        print(f"Model: {response.model_used}")
        print(f"Answer: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        
        await adapter.close()
        return True
        
    except Exception as e:
        print(f"❌ Answer Agent Error: {e}")
        return False


async def test_response_formatter_agent():
    """
    Test ResponseFormatterAgent - Merged Verifier + Response Agent.
    Handles: verification + formatting in single LLM call.
    """
    print("\n📝 Testing Response Formatter Agent (merged verifier + response)...")
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ API key not found!")
        return False
    
    try:
        adapter = OpenRouterAdapter(api_key=api_key, timeout=None)
        
        raw_answer = """Học phí tại UIT năm 2024 là 756.000 VNĐ/tín chỉ cho sinh viên công lập. 
Để đăng ký học phần, sinh viên cần đăng nhập vào hệ thống Portal vào thời gian quy định 
(tháng 7-8 cho học kỳ 1, tháng 12-1 cho học kỳ 2)."""

        rag_context = [
            "Học phí tại Đại học Công nghệ Thông tin (UIT) năm 2024: Sinh viên công lập: 756.000 VNĐ/tín chỉ.",
            "Thời gian đăng ký học phần: Thường vào tháng 7-8 cho học kỳ 1, tháng 12-1 cho học kỳ 2."
        ]
        
        context = ConversationContext(
            session_id="test-response-formatter",
            messages=[],
            system_prompt=f"""Bạn là ResponseFormatterAgent cho hệ thống Chatbot-UIT.
Nhiệm vụ của bạn là:
1. Xác minh câu trả lời với RAG context
2. Đánh giá độ tin cậy
3. Format response thân thiện, dễ hiểu

RAW ANSWER:
{raw_answer}

RAG CONTEXT:
{chr(10).join(rag_context)}

Trả lời bằng JSON format:
{{
    "verification_score": 1-10,
    "confidence": "high|medium|low",
    "final_response": "formatted response",
    "suggestions": ["optional follow-up suggestions"]
}}"""
        )
        
        request = AgentRequest(
            prompt="Xác minh và format câu trả lời về học phí UIT",
            context=context,
            model="deepseek/deepseek-r1:free",
            temperature=0.3,
            max_tokens=500
        )
        
        response = await adapter.generate_response(request)
        
        print(f"✅ Response Formatter Agent Response:")
        print(f"Model: {response.model_used}")
        print(f"Formatted Response: {response.content}")
        print(f"Tokens: {response.tokens_used}")
        
        await adapter.close()
        return True
        
    except Exception as e:
        print(f"❌ Response Formatter Agent Error: {e}")
        return False


async def main():
    """Run all optimized agent tests."""
    print("🚀 Testing Optimized 3-Agent Pipeline")
    print("=" * 60)
    print("Pipeline: SmartPlanner → Answer → ResponseFormatter")
    print("Cost Savings: ~40% fewer LLM calls compared to 5-agent pipeline")
    print("=" * 60)
    
    tests = [
        ("Smart Planner Agent", test_smart_planner_agent),
        ("Answer Agent", test_answer_agent),
        ("Response Formatter Agent", test_response_formatter_agent),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            print(f"\n{'='*20} {name} {'='*20}")
            result = await test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name} failed with exception: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS:")
    print("=" * 60)
    
    passed = 0
    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} agents working correctly")
    
    if passed == len(results):
        print("🎉 ALL OPTIMIZED AGENTS WORKING PERFECTLY!")
    else:
        print("⚠️  Some agents need attention.")


if __name__ == "__main__":
    asyncio.run(main())

"""
Agent implementations for the multi-agent orchestrator.

Optimized 3-Agent Pipeline:
- SmartPlannerAgent: Combined planning + query rewriting
- AnswerAgent: Answer generation from context
- ResponseFormatterAgent: Combined verification + formatting

Usage:
    from app.agents.optimized_orchestrator import OptimizedMultiAgentOrchestrator
"""

# Base classes
from .base import (
    SpecializedAgent,
    AgentConfig,
    AgentType,
    AnswerResult,
)

# Optimized agents (3-agent pipeline)
from .smart_planner_agent import SmartPlannerAgent, SmartPlanResult
from .answer_agent import AnswerAgent
from .response_formatter_agent import ResponseFormatterAgent, FormattedResponseResult

# Graph Reasoning Agent
from .graph_reasoning_agent import GraphReasoningAgent, GraphQueryType, GraphReasoningResult

# Orchestrator
from .optimized_orchestrator import OptimizedMultiAgentOrchestrator

__all__ = [
    # Base
    "SpecializedAgent",
    "AgentConfig", 
    "AgentType",
    "AnswerResult",
    
    # Optimized agents
    "SmartPlannerAgent",
    "SmartPlanResult",
    "AnswerAgent",
    "ResponseFormatterAgent",
    "FormattedResponseResult",
    
    # Graph Reasoning
    "GraphReasoningAgent",
    "GraphQueryType",
    "GraphReasoningResult",
    
    # Orchestrator
    "OptimizedMultiAgentOrchestrator",
]
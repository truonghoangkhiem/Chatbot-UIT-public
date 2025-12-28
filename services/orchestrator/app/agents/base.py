"""
Base classes and interfaces for specialized agents.

This module defines the base agent interface and common functionality
for all specialized agents in the multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

from ..core.domain import AgentRequest, AgentResponse
from ..ports.agent_ports import AgentPort


class AgentType(Enum):
    """Types of specialized agents (optimized 3-agent pipeline)."""
    SMART_PLANNER = "smart_planner"  # Planning + Query Rewriting
    ANSWER_AGENT = "answer_agent"  # Answer generation
    RESPONSE_FORMATTER = "response_formatter"  # Verification + Formatting


@dataclass
class AgentConfig:
    """Configuration for a specialized agent."""
    agent_type: AgentType
    model: str
    system_prompt: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    timeout: int = 30
    max_retries: int = 3
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class DetailedSource:
    """Detailed source information with citation precision."""
    title: str
    doc_id: Optional[str] = None
    chunk_id: Optional[str] = None
    score: float = 0.0
    # Citation details
    citation_text: Optional[str] = None  # Exact text quoted
    char_spans: Optional[List[Dict[str, Any]]] = None  # List of {start, end, text, type}
    highlighted_text: Optional[List[str]] = None  # Highlighted matches
    # Document metadata
    doc_type: Optional[str] = None
    faculty: Optional[str] = None
    year: Optional[int] = None
    subject: Optional[str] = None


@dataclass
class AnswerResult:
    """Result from answer agent with enhanced citation support."""
    query: str
    answer: str
    confidence: float
    sources_used: List[str]  # Simple list of titles (backward compatible)
    reasoning_steps: List[str]
    metadata: Dict[str, Any]
    # Enhanced citation data
    detailed_sources: List[DetailedSource] = field(default_factory=list)


class SpecializedAgent(ABC):
    """
    Base class for all specialized agents.
    
    This class provides common functionality and defines the interface
    that all specialized agents must implement.
    """
    
    def __init__(self, config: AgentConfig, agent_port: AgentPort):
        """
        Initialize the specialized agent.
        
        Args:
            config: Agent configuration
            agent_port: Port for communicating with the underlying LLM
        """
        self.config = config
        self.agent_port = agent_port
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and return results.
        
        Args:
            input_data: Input data specific to the agent type
            
        Returns:
            Processing results specific to the agent type
        """
        pass
    
    async def _make_agent_request(
        self, 
        prompt: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Make a request to the underlying agent.
        
        Args:
            prompt: The prompt to send to the agent
            context: Optional context information
            
        Returns:
            Agent response
        """
        # Import ConversationContext here to avoid circular dependency
        from ..core.domain import ConversationContext
        import logging
        import os
        
        logger = logging.getLogger(__name__)
        
        # Log input if debug mode is enabled
        if os.getenv('LOG_LEVEL', 'INFO').upper() == 'DEBUG':
            logger.debug(f"\n{'='*80}")
            logger.debug(f"🔵 AGENT INPUT - {self.config.agent_type.value.upper()}")
            logger.debug(f"{'='*80}")
            logger.debug(f"System Prompt Length: {len(self.config.system_prompt)} chars")
            logger.debug(f"System Prompt Preview: {self.config.system_prompt[:200]}...")
            logger.debug(f"User Prompt: {prompt[:500]}..." if len(prompt) > 500 else f"User Prompt: {prompt}")
            logger.debug(f"Model: {self.config.model}")
            logger.debug(f"Temperature: {self.config.temperature}")
            logger.debug(f"Max Tokens: {self.config.max_tokens}")
            if context:
                logger.debug(f"Context keys: {list(context.keys())}")
            logger.debug(f"{'='*80}\n")
        
        # Create conversation context with system prompt from agent config
        conversation_context = ConversationContext(
            session_id="agent_session",
            messages=[],
            system_prompt=self.config.system_prompt,  # ✅ FIX: Include system prompt!
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        request = AgentRequest(
            prompt=prompt,
            context=conversation_context,  # ✅ FIX: Pass context with system prompt!
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            metadata={"agent_type": self.config.agent_type.value}
        )
        
        response = await self.agent_port.generate_response(request)
        
        # Log output if debug mode is enabled
        if os.getenv('LOG_LEVEL', 'INFO').upper() == 'DEBUG':
            logger.debug(f"\n{'='*80}")
            logger.debug(f"🟢 AGENT OUTPUT - {self.config.agent_type.value.upper()}")
            logger.debug(f"{'='*80}")
            logger.debug(f"Response: {response.content[:500]}..." if len(response.content) > 500 else f"Response: {response.content}")
            logger.debug(f"Tokens Used: {response.tokens_used}")
            logger.debug(f"{'='*80}\n")
        
        return response
    
    def _build_system_prompt(self, user_prompt: str) -> str:
        """
        Build the complete system prompt.
        
        Args:
            user_prompt: User-specific prompt content
            
        Returns:
            Complete system prompt
        """
        return f"{self.config.system_prompt}\n\n{user_prompt}"
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about this agent."""
        return {
            "type": self.config.agent_type.value,
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens
        }
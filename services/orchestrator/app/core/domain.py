"""
Domain models for orchestrator service.

This module contains the core domain entities that represent the business
concepts in the orchestrator layer, independent of any external frameworks
or infrastructure concerns.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime


class ConversationRole(Enum):
    """Roles in a conversation."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class AgentProvider(Enum):
    """Supported agent providers."""
    OPENROUTER = "openrouter"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class MessageType(Enum):
    """Types of messages in the conversation."""
    TEXT = "text"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESULT = "function_result"


@dataclass
class ConversationMessage:
    """A single message in a conversation."""
    role: ConversationRole
    content: str
    timestamp: datetime
    message_type: MessageType = MessageType.TEXT
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "message_type": self.message_type.value,
            "metadata": self.metadata or {}
        }


@dataclass
class ConversationContext:
    """Context for a conversation session."""
    session_id: str
    messages: List[ConversationMessage]
    system_prompt: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    metadata: Optional[Dict[str, Any]] = None
    
    def add_message(self, message: ConversationMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
    
    def get_recent_messages(self, count: int) -> List[ConversationMessage]:
        """Get the most recent N messages."""
        return self.messages[-count:] if count < len(self.messages) else self.messages


@dataclass
class AgentRequest:
    """Request to an agent service."""
    prompt: str
    context: Optional[ConversationContext] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: bool = False
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AgentResponse:
    """Response from an agent service."""
    content: str
    model_used: str
    tokens_used: Optional[int] = None
    finish_reason: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RAGContext:
    """Context from RAG system to be used in agent requests."""
    query: str
    retrieved_documents: List[Dict[str, Any]]
    search_metadata: Optional[Dict[str, Any]] = None
    relevance_scores: Optional[List[float]] = None
    rewritten_queries: Optional[List[str]] = None  # Queries from query rewriter


@dataclass
class OrchestrationRequest:
    """Complete request for the orchestration pipeline."""
    user_query: str
    session_id: Optional[str] = None
    use_rag: bool = True
    use_knowledge_graph: Optional[bool] = None  # Force KG usage (overrides SmartPlanner)
    rag_top_k: int = 5
    agent_model: Optional[str] = None
    conversation_context: Optional[ConversationContext] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OrchestrationResponse:
    """Complete response from the orchestration pipeline."""
    response: str
    session_id: str
    rag_context: Optional[RAGContext] = None
    agent_metadata: Optional[Dict[str, Any]] = None
    processing_stats: Optional[Dict[str, Any]] = None
    timestamp: Optional[datetime] = None
"""
LLM Client Interface and Base Implementation.

This module provides an abstract interface for LLM clients and common utilities
for working with different LLM providers.

Week 2 - Task B1: LLM Relation Extraction
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class LLMProvider(str, Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    GEMINI = "gemini"
    OLLAMA = "ollama"  # Local LLM
    ANTHROPIC = "anthropic"


@dataclass
class LLMResponse:
    """Response from LLM API"""
    text: str
    model: str
    provider: LLMProvider
    tokens_used: int = 0
    cost_usd: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "model": self.model,
            "provider": self.provider.value,
            "tokens_used": self.tokens_used,
            "cost_usd": self.cost_usd,
            "metadata": self.metadata,
        }


@dataclass
class LLMConfig:
    """Configuration for LLM client"""
    api_key: Optional[str] = None
    model: str = "gpt-4"
    temperature: float = 0.1
    max_tokens: int = 2000
    timeout: int = 30
    retry_count: int = 3
    base_url: Optional[str] = None  # For custom endpoints
    
    # Cost tracking
    track_costs: bool = True
    cost_per_1k_input_tokens: float = 0.03  # GPT-4 default
    cost_per_1k_output_tokens: float = 0.06
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }


class LLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    Provides a unified interface for different LLM providers.
    
    Example:
        ```python
        client = OpenAIClient(api_key="sk-...")
        
        response = await client.complete(
            prompt="Trích xuất các quan hệ từ văn bản sau...",
            temperature=0.1,
            max_tokens=1000
        )
        
        print(response.text)
        ```
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialize LLM client.
        
        Args:
            config: LLM configuration
        """
        self.config = config
        self._total_tokens = 0
        self._total_cost = 0.0
    
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Complete a prompt.
        
        Args:
            prompt: Input prompt text
            temperature: Sampling temperature (0-1, lower = more deterministic)
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse with generated text
        """
        pass
    
    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Chat completion with message history.
        
        Args:
            messages: List of messages [{"role": "user", "content": "..."}]
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Provider-specific parameters
            
        Returns:
            LLMResponse with generated text
        """
        pass
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """
        Get usage statistics.
        
        Returns:
            Dictionary with tokens and cost information
        """
        return {
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 4),
            "average_tokens_per_request": (
                self._total_tokens / self._request_count if hasattr(self, '_request_count') else 0
            ),
        }
    
    def reset_stats(self):
        """Reset usage statistics"""
        self._total_tokens = 0
        self._total_cost = 0.0
        self._request_count = 0
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """
        Calculate API cost.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            
        Returns:
            Cost in USD
        """
        if not self.config.track_costs:
            return 0.0
        
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input_tokens
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output_tokens
        
        return input_cost + output_cost
    
    def _update_stats(self, tokens: int, cost: float):
        """Update usage statistics"""
        self._total_tokens += tokens
        self._total_cost += cost
        
        if not hasattr(self, '_request_count'):
            self._request_count = 0
        self._request_count += 1


class MockLLMClient(LLMClient):
    """
    Mock LLM client for testing.
    
    Returns predefined responses without calling actual LLM APIs.
    """
    
    def __init__(self, mock_responses: Optional[List[str]] = None):
        """
        Initialize mock client.
        
        Args:
            mock_responses: Predefined responses to return
        """
        config = LLMConfig(model="mock", api_key="mock")
        super().__init__(config)
        
        self.mock_responses = mock_responses or [
            '{"entities": [], "relations": []}'
        ]
        self._response_index = 0
    
    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Return mock response"""
        response_text = self.mock_responses[
            self._response_index % len(self.mock_responses)
        ]
        self._response_index += 1
        
        return LLMResponse(
            text=response_text,
            model="mock",
            provider=LLMProvider.OPENAI,
            tokens_used=len(prompt.split()) + len(response_text.split()),
        )
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Return mock response"""
        # Combine all messages as prompt
        prompt = "\n".join([m.get("content", "") for m in messages])
        return await self.complete(prompt, temperature, max_tokens)

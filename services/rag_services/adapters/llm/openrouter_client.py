"""
OpenRouter Client Implementation.

OpenRouter provides unified access to multiple LLM providers including:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini)
- Meta (Llama)
- Mistral
- And many more

This is perfect for flexible model selection without changing code.
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from .llm_client import LLMClient, LLMResponse, LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class OpenRouterClient(LLMClient):
    """
    OpenRouter client implementation.
    
    Provides unified access to multiple LLM providers through OpenRouter API.
    Uses OpenAI-compatible API format.
    
    Example:
        ```python
        client = OpenRouterClient(
            api_key="sk-or-v1-...",
            model="google/gemini-flash-1.5"  # or any OpenRouter model
        )
        
        response = await client.complete(
            prompt="Trích xuất entities từ: IT003 cần hoàn thành IT002",
            temperature=0.1
        )
        ```
    
    Popular models:
    - google/gemini-flash-1.5 (fast, cheap)
    - google/gemini-pro-1.5 (balanced)
    - anthropic/claude-3.5-sonnet (high quality)
    - meta-llama/llama-3.1-70b-instruct (open source)
    - openai/gpt-4-turbo (OpenAI via OpenRouter)
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "google/gemini-flash-1.5",
        base_url: str = "https://openrouter.ai/api/v1",
        **kwargs
    ):
        """
        Initialize OpenRouter client.
        
        Args:
            api_key: OpenRouter API key (get from https://openrouter.ai)
            model: Model identifier (e.g., "google/gemini-flash-1.5")
            base_url: OpenRouter API base URL
            **kwargs: Additional config parameters
        """
        config = LLMConfig(
            api_key=api_key,
            model=model,
            base_url=base_url,
            **kwargs
        )
        super().__init__(config)
        
        # Update cost tracking (approximate for various models)
        # OpenRouter shows exact costs in response
        if "gemini" in model.lower():
            self.config.cost_per_1k_input_tokens = 0.000075
            self.config.cost_per_1k_output_tokens = 0.0003
        elif "claude" in model.lower():
            self.config.cost_per_1k_input_tokens = 0.003
            self.config.cost_per_1k_output_tokens = 0.015
        elif "gpt-4" in model.lower():
            self.config.cost_per_1k_input_tokens = 0.03
            self.config.cost_per_1k_output_tokens = 0.06
        elif "llama" in model.lower():
            self.config.cost_per_1k_input_tokens = 0.0007
            self.config.cost_per_1k_output_tokens = 0.0008
        
        # Initialize OpenAI client (OpenRouter uses OpenAI-compatible API)
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url
            )
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )
        
        logger.info(f"Initialized OpenRouter client with model: {model}")
    
    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Complete a prompt using OpenRouter.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated text
        """
        # Convert to chat format (required for most models)
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        return await self.chat(messages, temperature, max_tokens, **kwargs)
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Chat completion using OpenRouter.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated text
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Remove 'model' from kwargs if present to avoid duplicate parameter
        kwargs.pop('model', None)
        
        try:
            # Call OpenRouter API (OpenAI-compatible)
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            
            # Extract response
            text = response.choices[0].message.content
            
            # Calculate usage and cost
            usage = response.usage
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            cost = self._calculate_cost(input_tokens, output_tokens)
            self._update_stats(total_tokens, cost)
            
            logger.info(
                f"OpenRouter API call ({self.config.model}): "
                f"{total_tokens} tokens, ${cost:.4f}"
            )
            
            return LLMResponse(
                text=text,
                model=self.config.model,
                provider=LLMProvider.OPENAI,  # Compatible API
                tokens_used=total_tokens,
                cost_usd=cost,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                    "provider": "openrouter",
                }
            )
            
        except Exception as e:
            logger.error(f"OpenRouter API call failed: {e}")
            
            # Retry logic
            for retry in range(self.config.retry_count):
                logger.info(f"Retrying... ({retry + 1}/{self.config.retry_count})")
                await asyncio.sleep(2 ** retry)  # Exponential backoff
                
                try:
                    return await self.chat(messages, temperature, max_tokens, **kwargs)
                except Exception as retry_error:
                    if retry == self.config.retry_count - 1:
                        logger.error(f"All retries failed: {retry_error}")
                        raise
            
            raise


def create_llm_client_from_env() -> LLMClient:
    """
    Create LLM client from environment variables.
    
    Reads:
    - LLM_PROVIDER: openai, gemini, openrouter
    - OPENAI_API_KEY: For OpenAI or OpenRouter (when provider=openrouter)
    - OPENAI_BASE_URL: For OpenRouter (https://openrouter.ai/api/v1)
    - LLM_MODEL: Model identifier
    - GEMINI_API_KEY: For direct Gemini
    
    Returns:
        Configured LLM client
    """
    import os
    
    provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
    model = os.getenv("LLM_MODEL", "google/gemini-flash-1.5")
    
    if provider == "openrouter":
        api_key = os.getenv("OPENAI_API_KEY")  # OpenRouter uses this field
        base_url = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1")
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set in environment. "
                "For OpenRouter, set OPENAI_API_KEY to your OpenRouter key (sk-or-v1-...)"
            )
        
        return OpenRouterClient(
            api_key=api_key,
            model=model,
            base_url=base_url
        )
    
    elif provider == "openai":
        from .openai_client import OpenAIClient
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in environment")
        
        return OpenAIClient(
            api_key=api_key,
            model=model
        )
    
    elif provider == "gemini":
        from .openai_client import GeminiClient
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set in environment")
        
        return GeminiClient(
            api_key=api_key,
            model=model
        )
    
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")

"""
OpenAI Client Implementation.

Wrapper for OpenAI GPT models (GPT-4, GPT-3.5, etc.)
"""

import logging
from typing import List, Dict, Any, Optional
import asyncio

from .llm_client import LLMClient, LLMResponse, LLMConfig, LLMProvider

logger = logging.getLogger(__name__)


class OpenAIClient(LLMClient):
    """
    OpenAI GPT client implementation.
    
    Supports GPT-4, GPT-4-turbo, GPT-3.5-turbo, etc.
    
    Example:
        ```python
        client = OpenAIClient(
            api_key="sk-...",
            model="gpt-4-turbo-preview"
        )
        
        response = await client.complete(
            prompt="Trích xuất entities từ: IT003 cần hoàn thành IT002",
            temperature=0.1
        )
        ```
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4-turbo-preview",
        **kwargs
    ):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-4-turbo-preview, gpt-3.5-turbo, etc.)
            **kwargs: Additional config parameters
        """
        config = LLMConfig(
            api_key=api_key,
            model=model,
            **kwargs
        )
        super().__init__(config)
        
        # Update cost tracking for specific models
        if "gpt-4" in model:
            self.config.cost_per_1k_input_tokens = 0.03
            self.config.cost_per_1k_output_tokens = 0.06
        elif "gpt-3.5" in model:
            self.config.cost_per_1k_input_tokens = 0.0015
            self.config.cost_per_1k_output_tokens = 0.002
        
        # Initialize OpenAI client
        try:
            from openai import AsyncOpenAI
            self.client = AsyncOpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "OpenAI package not installed. Run: pip install openai"
            )
    
    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Complete a prompt using OpenAI.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature (0-1)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLMResponse with generated text
        """
        # Convert to chat format (required for GPT-4)
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
        Chat completion using OpenAI.
        
        Args:
            messages: Chat messages
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional OpenAI parameters
            
        Returns:
            LLMResponse with generated text
        """
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        # Remove 'model' from kwargs if present to avoid duplicate parameter
        kwargs.pop('model', None)
        
        try:
            # Call OpenAI API
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
                f"OpenAI API call: {total_tokens} tokens, ${cost:.4f}"
            )
            
            return LLMResponse(
                text=text,
                model=self.config.model,
                provider=LLMProvider.OPENAI,
                tokens_used=total_tokens,
                cost_usd=cost,
                metadata={
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "finish_reason": response.choices[0].finish_reason,
                }
            )
            
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            
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


class GeminiClient(LLMClient):
    """
    Google Gemini client implementation.
    
    Fallback option for OpenAI.
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-pro",
        **kwargs
    ):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google API key
            model: Model name (gemini-pro, gemini-pro-vision, etc.)
            **kwargs: Additional config parameters
        """
        config = LLMConfig(
            api_key=api_key,
            model=model,
            cost_per_1k_input_tokens=0.00025,  # Gemini pricing
            cost_per_1k_output_tokens=0.0005,
            **kwargs
        )
        super().__init__(config)
        
        # Initialize Gemini client
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model)
        except ImportError:
            raise ImportError(
                "Google AI package not installed. Run: pip install google-generativeai"
            )
    
    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Complete using Gemini"""
        temperature = temperature or self.config.temperature
        max_tokens = max_tokens or self.config.max_tokens
        
        try:
            # Call Gemini API
            response = await asyncio.to_thread(
                self.client.generate_content,
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )
            
            text = response.text
            
            # Estimate tokens (Gemini doesn't provide exact count)
            estimated_tokens = len(prompt.split()) + len(text.split())
            cost = self._calculate_cost(
                len(prompt.split()),
                len(text.split())
            )
            self._update_stats(estimated_tokens, cost)
            
            return LLMResponse(
                text=text,
                model=self.config.model,
                provider=LLMProvider.GEMINI,
                tokens_used=estimated_tokens,
                cost_usd=cost,
            )
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            raise
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Chat using Gemini (convert to single prompt)"""
        # Combine messages into single prompt
        prompt = "\n\n".join([
            f"{m['role']}: {m['content']}"
            for m in messages
        ])
        
        return await self.complete(prompt, temperature, max_tokens)

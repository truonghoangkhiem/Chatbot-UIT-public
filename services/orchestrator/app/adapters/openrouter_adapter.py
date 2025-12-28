"""
OpenRouter API adapter implementation.

This adapter provides integration with OpenRouter API following the
Ports & Adapters architecture pattern.
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Optional, AsyncGenerator, Dict, Any
from datetime import datetime

from ..ports.agent_ports import AgentPort
from ..core.domain import (
    AgentRequest, 
    AgentResponse, 
    ConversationContext,
    ConversationMessage,
    ConversationRole,
    AgentProvider,
    MessageType
)


class OpenRouterAdapter(AgentPort):
    """
    Adapter for OpenRouter API integration.
    
    This adapter implements the AgentPort interface to provide
    communication with OpenRouter's LLM services.
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        default_model: str = "google/gemma-3-27b-it:free",
        timeout: Optional[int] = 30,
        max_retries: int = 3
    ):
        """
        Initialize the OpenRouter adapter.
        
        Args:
            api_key: OpenRouter API key
            base_url: Base URL for OpenRouter API
            default_model: Default model to use if none specified
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.api_key = api_key
        self.base_url = base_url
        self.default_model = default_model
        self.timeout = timeout
        self.max_retries = max_retries
        
        self._session: Optional[aiohttp.ClientSession] = None
        self._supported_models: Optional[List[str]] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            # Prepare session kwargs
            session_kwargs = {
                "headers": {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://localhost:3000",  # Required by OpenRouter
                    "X-Title": "Chatbot-UIT"  # App identification
                }
            }
            
            # Only add timeout if specified (None = no timeout)
            if self.timeout is not None:
                session_kwargs["timeout"] = aiohttp.ClientTimeout(total=self.timeout)
            
            self._session = aiohttp.ClientSession(**session_kwargs)
        return self._session
    
    def _prepare_messages(self, request: AgentRequest) -> List[Dict[str, str]]:
        """
        Prepare messages for OpenRouter API format.
        
        Args:
            request: The agent request
            
        Returns:
            List of messages in OpenRouter format
        """
        messages = []
        
        # Add system message if context has system prompt
        if request.context and request.context.system_prompt:
            messages.append({
                "role": "system",
                "content": request.context.system_prompt
            })
        
        # Add conversation history if available
        if request.context and request.context.messages:
            for msg in request.context.messages[-10:]:  # Limit to last 10 messages
                if msg.message_type == MessageType.TEXT:
                    messages.append({
                        "role": msg.role.value,
                        "content": msg.content
                    })
        
        # Add the current prompt
        messages.append({
            "role": "user",
            "content": request.prompt
        })
        
        return messages
    
    def _prepare_request_payload(self, request: AgentRequest) -> Dict[str, Any]:
        """
        Prepare the complete request payload for OpenRouter API.
        
        Args:
            request: The agent request
            
        Returns:
            Dictionary containing the request payload
        """
        messages = self._prepare_messages(request)
        
        payload = {
            "model": request.model or self.default_model,
            "messages": messages,
            "stream": request.stream
        }
        
        # Add optional parameters
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        
        # Add metadata if available
        if request.metadata:
            # OpenRouter supports some additional parameters
            if "top_p" in request.metadata:
                payload["top_p"] = request.metadata["top_p"]
            if "frequency_penalty" in request.metadata:
                payload["frequency_penalty"] = request.metadata["frequency_penalty"]
            if "presence_penalty" in request.metadata:
                payload["presence_penalty"] = request.metadata["presence_penalty"]
        
        return payload
    
    async def generate_response(self, request: AgentRequest) -> AgentResponse:
        """
        Generate a response from OpenRouter API.
        
        Args:
            request: The agent request
            
        Returns:
            AgentResponse containing the generated response
        """
        session = await self._get_session()
        payload = self._prepare_request_payload(request)
        payload["stream"] = False  # Ensure non-streaming for this method
        
        start_time = time.time()
        
        for attempt in range(self.max_retries):
            try:
                async with session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract response content
                        content = data["choices"][0]["message"]["content"]
                        model_used = data.get("model", payload["model"])
                        finish_reason = data["choices"][0].get("finish_reason")
                        
                        # Extract token usage if available
                        tokens_used = None
                        if "usage" in data:
                            tokens_used = data["usage"].get("total_tokens")
                        
                        processing_time = time.time() - start_time
                        
                        return AgentResponse(
                            content=content,
                            model_used=model_used,
                            tokens_used=tokens_used,
                            finish_reason=finish_reason,
                            processing_time=processing_time,
                            metadata={
                                "provider": "openrouter",
                                "api_response": data,
                                "attempt": attempt + 1
                            }
                        )
                    
                    elif response.status == 429:  # Rate limit
                        if attempt < self.max_retries - 1:
                            wait_time = 2 ** attempt  # Exponential backoff
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise Exception(f"Rate limit exceeded after {self.max_retries} attempts")
                    
                    else:
                        error_text = await response.text()
                        raise Exception(f"OpenRouter API error {response.status}: {error_text}")
            
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Network error after {self.max_retries} attempts: {str(e)}")
        
        raise Exception("Max retries exceeded")
    
    async def stream_response(self, request: AgentRequest) -> AsyncGenerator[str, None]:
        """
        Stream a response from OpenRouter API.
        
        Args:
            request: The agent request
            
        Yields:
            String chunks of the response
        """
        session = await self._get_session()
        payload = self._prepare_request_payload(request)
        payload["stream"] = True
        
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API error {response.status}: {error_text}")
                
                async for line in response.content:
                    line_str = line.decode('utf-8').strip()
                    
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # Remove 'data: ' prefix
                        
                        if data_str == '[DONE]':
                            break
                        
                        try:
                            data = json.loads(data_str)
                            
                            if 'choices' in data and len(data['choices']) > 0:
                                delta = data['choices'][0].get('delta', {})
                                
                                if 'content' in delta:
                                    yield delta['content']
                        
                        except json.JSONDecodeError:
                            # Skip malformed JSON lines
                            continue
        
        except aiohttp.ClientError as e:
            raise Exception(f"Streaming error: {str(e)}")
    
    async def validate_connection(self) -> bool:
        """
        Validate connection to OpenRouter API.
        
        Returns:
            True if connection is valid, False otherwise
        """
        try:
            session = await self._get_session()
            
            # Make a simple request to validate the connection
            test_payload = {
                "model": self.default_model,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 1
            }
            
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=test_payload
            ) as response:
                return response.status in [200, 429]  # 429 means rate limit but connection works
        
        except Exception:
            return False
    
    def get_supported_models(self) -> List[str]:
        """
        Get supported models from OpenRouter.
        
        Note: This returns a static list. In production, you might want to
        fetch this dynamically from the OpenRouter models endpoint.
        """
        if self._supported_models is None:
            # Common OpenRouter models - in production, fetch from API
            self._supported_models = [
                "google/gemma-3-27b-it:free",

            ]
        
        return self._supported_models.copy()
    
    async def generate(self, messages: list, temperature: float = 0.7, max_tokens: int = 1000, model: Optional[str] = None) -> Any:
        """
        Generate a response using the chat completion endpoint.
        Alias method to support GraphReasoningAgent interface.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model to use (defaults to self.default_model)
            
        Returns:
            Response object with .content attribute containing the generated text
        """
        # Create AgentRequest from raw parameters
        request = AgentRequest(
            prompt="",  # Prompt is in messages
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
            context=None  # Messages will be added directly
        )
        
        # Override message preparation to use provided messages
        session = await self._get_session()
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data["choices"][0]["message"]["content"]
                    
                    # Return object with .content attribute for compatibility
                    class ResponseWrapper:
                        def __init__(self, content):
                            self.content = content
                    
                    return ResponseWrapper(content)
                else:
                    error_text = await response.text()
                    raise Exception(f"OpenRouter API error: {response.status} - {error_text}")
        except Exception as e:
            raise Exception(f"Failed to generate response: {str(e)}")
    
    def get_provider_info(self) -> AgentProvider:
        """Get provider information."""
        return AgentProvider.OPENROUTER
    
    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
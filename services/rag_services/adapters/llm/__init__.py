"""
LLM adapters for graph building and relation extraction.

This package provides LLM client wrappers for:
- OpenAI GPT-4
- Google Gemini
- OpenRouter (unified access to multiple providers)
- Local LLMs (via Ollama)
"""

from .llm_client import LLMClient, LLMResponse, LLMProvider
from .openai_client import OpenAIClient, GeminiClient
from .openrouter_client import OpenRouterClient, create_llm_client_from_env

__all__ = [
    'LLMClient',
    'LLMResponse',
    'LLMProvider',
    'OpenAIClient',
    'GeminiClient',
    'OpenRouterClient',
    'create_llm_client_from_env',
]

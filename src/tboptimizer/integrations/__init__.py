"""LLM integrations with bandwidth optimization."""

from .claude_adapter import ClaudeOptimizedClient, CollaborationContext, OptimizedResponse
from .openai_adapter import OpenAIOptimizedClient
from .generic_llm import GenericLLMClient, LLMProvider, CustomAPIProvider, LocalModelProvider

__all__ = [
    "ClaudeOptimizedClient",
    "OpenAIOptimizedClient", 
    "GenericLLMClient",
    "CollaborationContext",
    "OptimizedResponse",
    "LLMProvider",
    "CustomAPIProvider",
    "LocalModelProvider",
]
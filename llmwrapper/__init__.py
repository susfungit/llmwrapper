"""
LLM Wrapper - A vendor-agnostic Python wrapper for multiple Large Language Models.

This package provides a unified interface to interact with:
- OpenAI GPT models
- Anthropic Claude models
- Google Gemini models
- Grok (xAI) models

Both synchronous and asynchronous operations are supported.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main classes and functions
from .factory import get_llm
from .async_factory import get_async_llm
from .base import BaseLLM
from .async_base import AsyncBaseLLM

# Import all wrapper classes
from .openai_wrapper import OpenAIWrapper
from .anthropic_wrapper import ClaudeWrapper
from .gemini_wrapper import GeminiWrapper
from .grok_wrapper import GrokWrapper

# Import async wrapper classes
from .async_openai_wrapper import AsyncOpenAIWrapper
from .async_anthropic_wrapper import AsyncClaudeWrapper
from .async_gemini_wrapper import AsyncGeminiWrapper
from .async_grok_wrapper import AsyncGrokWrapper

# Import utilities
from .logger import logger
from .logging_mixin import LoggingMixin

__all__ = [
    # Version info
    "__version__",
    
    # Main factory functions
    "get_llm",
    "get_async_llm",
    
    # Base classes
    "BaseLLM",
    "AsyncBaseLLM",
    
    # Sync wrapper classes
    "OpenAIWrapper",
    "ClaudeWrapper", 
    "GeminiWrapper",
    "GrokWrapper",
    
    # Async wrapper classes
    "AsyncOpenAIWrapper",
    "AsyncClaudeWrapper",
    "AsyncGeminiWrapper", 
    "AsyncGrokWrapper",
    
    # Utilities
    "logger",
    "LoggingMixin",
] 
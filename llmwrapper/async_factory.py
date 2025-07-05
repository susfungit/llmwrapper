from .registry import llm_registry
from .logger import logger

# Import all async providers to ensure they're registered
from .async_openai_wrapper import AsyncOpenAIWrapper
from .async_anthropic_wrapper import AsyncClaudeWrapper
from .async_gemini_wrapper import AsyncGeminiWrapper
from .async_grok_wrapper import AsyncGrokWrapper
from .async_ollama_wrapper import AsyncOllamaWrapper

def get_async_llm(provider: str, config: dict):
    """
    Factory function to create async LLM wrapper instances using the registry pattern.
    
    Args:
        provider: Name of the LLM provider ('openai', 'anthropic', 'gemini', 'grok', 'ollama')
        config: Configuration dictionary containing API keys and other settings
        
    Returns:
        AsyncBaseLLM: An instance of the appropriate async LLM wrapper
        
    Raises:
        ValueError: If the provider is not supported
    """
    try:
        return llm_registry.create_async_llm(provider, config)
    except ValueError as e:
        logger.error(f"Failed to create async LLM instance: {e}")
        raise 
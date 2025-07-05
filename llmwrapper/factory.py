from .registry import llm_registry
from .logger import logger

# Import all providers to ensure they're registered
from .openai_wrapper import OpenAIWrapper
from .anthropic_wrapper import ClaudeWrapper
from .gemini_wrapper import GeminiWrapper
from .grok_wrapper import GrokWrapper

def get_llm(provider: str, config: dict):
    """
    Factory function to create LLM wrapper instances using the registry pattern.
    
    Args:
        provider: Name of the LLM provider ('openai', 'anthropic', 'gemini', 'grok')
        config: Configuration dictionary containing API keys and other settings
        
    Returns:
        BaseLLM: An instance of the appropriate LLM wrapper
        
    Raises:
        ValueError: If the provider is not supported
    """
    try:
        return llm_registry.create_sync_llm(provider, config)
    except ValueError as e:
        logger.error(f"Failed to create LLM instance: {e}")
        raise
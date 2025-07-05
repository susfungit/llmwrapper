from .async_openai_wrapper import AsyncOpenAIWrapper
from .async_anthropic_wrapper import AsyncClaudeWrapper
from .async_gemini_wrapper import AsyncGeminiWrapper
from .async_grok_wrapper import AsyncGrokWrapper
from .logger import logger

def get_async_llm(provider: str, config: dict):
    """
    Factory function to create async LLM wrapper instances.
    
    Args:
        provider: Name of the LLM provider ('openai', 'anthropic', 'gemini', 'grok')
        config: Configuration dictionary containing API keys and other settings
        
    Returns:
        AsyncBaseLLM: An instance of the appropriate async LLM wrapper
        
    Raises:
        ValueError: If the provider is not supported
    """
    if provider == "openai":
        logger.info("Instantiating AsyncOpenAIWrapper")
        return AsyncOpenAIWrapper(api_key=config["api_key"], model=config.get("model", "gpt-4"))
    elif provider == "anthropic":
        logger.info("Instantiating AsyncClaudeWrapper")
        return AsyncClaudeWrapper(api_key=config["api_key"], model=config.get("model", "claude-3-opus-20240229"))
    elif provider == "gemini":
        logger.info("Instantiating AsyncGeminiWrapper")
        return AsyncGeminiWrapper(api_key=config["api_key"], model=config.get("model", "gemini-pro"))
    elif provider == "grok":
        logger.info("Instantiating AsyncGrokWrapper")
        return AsyncGrokWrapper(
            api_key=config["api_key"], 
            model=config.get("model", "grok-beta"),
            base_url=config.get("base_url", "https://api.x.ai/v1")
        )
    else:
        logger.error(f"Unsupported async LLM provider: {provider}")
        raise ValueError(f"Unsupported async LLM provider: {provider}") 
from openai_wrapper import OpenAIWrapper
from anthropic_wrapper import ClaudeWrapper
from gemini_wrapper import GeminiWrapper
from grok_wrapper import GrokWrapper
from logger import logger

def get_llm(provider: str, config: dict):
    if provider == "openai":
        logger.info("Instantiating OpenAIWrapper")
        return OpenAIWrapper(api_key=config["api_key"], model=config.get("model", "gpt-4"))
    elif provider == "anthropic":
        logger.info("Instantiating ClaudeWrapper")
        return ClaudeWrapper(api_key=config["api_key"], model=config.get("model", "claude-3-opus-20240229"))
    elif provider == "gemini":
        logger.info("Instantiating GeminiWrapper")
        return GeminiWrapper(api_key=config["api_key"], model=config.get("model", "gemini-pro"))
    elif provider == "grok":
        logger.info("Instantiating GrokWrapper")
        return GrokWrapper(api_key=config["api_key"], model=config.get("model", "grok-1"))
    else:
        logger.error(f"Unsupported LLM provider: {provider}")
        raise ValueError(f"Unsupported LLM provider: {provider}")
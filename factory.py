from .openai_wrapper import OpenAIWrapper
from .anthropic_wrapper import ClaudeWrapper

def get_llm(provider: str, config: dict):
    if provider == "openai":
        return OpenAIWrapper(api_key=config["api_key"], model=config.get("model", "gpt-4"))
    elif provider == "anthropic":
        return ClaudeWrapper(api_key=config["api_key"], model=config.get("model", "claude-3-opus-20240229"))
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
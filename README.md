# LLM Wrapper

A vendor-agnostic Python wrapper for interacting with multiple Large Language Models (LLMs) including OpenAI, Anthropic Claude, Google Gemini, and Grok (xAI).

## 🚀 Features

- **Unified Interface**: Single API to interact with multiple LLM providers
- **Easy Provider Switching**: Change providers with minimal code changes
- **Extensible**: Easy to add new providers by extending the base class
- **Type Safety**: Full type hints for better development experience

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from llmwrapper.factory import get_llm

# OpenAI Example
config = {
    "api_key": "your-openai-api-key",
    "model": "gpt-4"
}

llm = get_llm("openai", config)
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
])
print(response)
```

## 📦 Supported Providers

| Provider | Status | Default Model | Notes |
|----------|--------|---------------|-------|
| **OpenAI** | ✅ Active | `gpt-4` | Full support |
| **Anthropic** | ✅ Active | `claude-3-opus-20240229` | Full support |
| **Google Gemini** | ✅ Active | `gemini-pro` | Full support |
| **Grok (xAI)** | ⚠️ Placeholder | `grok-1` | API not publicly available |

## 🔧 Provider Configuration

### OpenAI
```python
config = {
    "api_key": "your-openai-api-key",
    "model": "gpt-4"  # or "gpt-3.5-turbo", "gpt-4-turbo", etc.
}
llm = get_llm("openai", config)
```

### Anthropic Claude
```python
config = {
    "api_key": "your-anthropic-api-key",
    "model": "claude-3-opus-20240229"  # or "claude-3-sonnet-20240229", etc.
}
llm = get_llm("anthropic", config)
```

### Google Gemini
```python
config = {
    "api_key": "your-google-api-key",
    "model": "gemini-pro"  # or "gemini-pro-vision"
}
llm = get_llm("gemini", config)
```

### Grok (xAI) - Placeholder
```python
# Note: Grok API is not publicly available yet
config = {
    "api_key": "your-grok-api-key",
    "model": "grok-1"
}
# This will raise NotImplementedError
llm = get_llm("grok", config)
```

## 📖 Usage Examples

### Basic Chat
```python
from llmwrapper.factory import get_llm

# Initialize any provider
llm = get_llm("anthropic", {
    "api_key": "your-api-key",
    "model": "claude-3-sonnet-20240229"
})

# Single message
response = llm.chat([
    {"role": "user", "content": "Explain quantum computing"}
])

# Multi-turn conversation
messages = [
    {"role": "system", "content": "You are a Python expert."},
    {"role": "user", "content": "How do I create a class in Python?"},
    {"role": "assistant", "content": "You create a class using the 'class' keyword..."},
    {"role": "user", "content": "Can you show me an example?"}
]
response = llm.chat(messages)
```

### Provider Switching
```python
# Easy to switch between providers
providers = ["openai", "anthropic", "gemini"]
configs = {
    "openai": {"api_key": "openai_key", "model": "gpt-4"},
    "anthropic": {"api_key": "anthropic_key", "model": "claude-3-opus-20240229"},
    "gemini": {"api_key": "gemini_key", "model": "gemini-pro"}
}

for provider in providers:
    llm = get_llm(provider, configs[provider])
    response = llm.chat([{"role": "user", "content": "Hello!"}])
    print(f"{provider}: {response}")
```

## 🏗️ Architecture

### Project Structure
```
llmwrapper/
├── base.py                 # Abstract base class
├── openai_wrapper.py       # OpenAI implementation
├── anthropic_wrapper.py    # Anthropic Claude implementation
├── gemini_wrapper.py       # Google Gemini implementation
├── grok_wrapper.py         # Grok placeholder implementation
├── factory.py              # Factory pattern for provider selection
├── example_usage.py        # Usage examples
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Adding New Providers

To add a new LLM provider:

1. Create a new wrapper class extending `BaseLLM`:
```python
from .base import BaseLLM

class NewProviderWrapper(BaseLLM):
    def __init__(self, api_key: str, model: str):
        # Initialize your provider
        pass
    
    def chat(self, messages: list[dict], **kwargs) -> str:
        # Implement chat functionality
        pass
```

2. Update `factory.py` to include your provider:
```python
from .new_provider_wrapper import NewProviderWrapper

def get_llm(provider: str, config: dict):
    # ... existing providers ...
    elif provider == "new_provider":
        return NewProviderWrapper(api_key=config["api_key"], model=config.get("model", "default-model"))
```

## 🔐 API Keys Setup

You'll need API keys from the respective providers:

- **OpenAI**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic**: Get from [Anthropic Console](https://console.anthropic.com/)
- **Google Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Grok**: Not publicly available yet

## ⚠️ Important Notes

1. **Grok Implementation**: The Grok wrapper is currently a placeholder as the xAI APIs are not publicly available.

2. **Rate Limits**: Each provider has different rate limits. Implement appropriate rate limiting in production.

3. **Error Handling**: The current implementation provides basic error handling. Consider adding more robust error handling for production use.

4. **Security**: Never commit API keys to version control. Use environment variables or secure configuration files.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.
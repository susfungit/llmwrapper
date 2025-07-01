# LLM Wrapper

A vendor-agnostic Python wrapper for interacting with multiple Large Language Models (LLMs) including OpenAI, Anthropic Claude, Google Gemini, and Grok (xAI).

## 🚀 Features

- **Unified Interface**: Single API to interact with multiple LLM providers
- **Easy Provider Switching**: Change providers with minimal code changes
- **Extensible**: Easy to add new providers by extending the base class
- **Type Safety**: Full type hints for better development experience
- **Enhanced Logging**: Comprehensive logging with provider/model identification, timing, and token usage
- **Provider Initialization Logging**: Track wrapper instantiation and configuration
- **Comprehensive Testing**: Full test coverage with pytest and advanced mocking
- **Secure Configuration**: Environment variable and config file support
- **Modern API Support**: Uses latest OpenAI SDK v1.0.0+ and Google Gemini API

## 🔧 Installation

```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

```python
from factory import get_llm

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
| **OpenAI** | ✅ Active | `gpt-4` | Full support with logging, modern SDK v1.0.0+ |
| **Anthropic** | ✅ Active | `claude-3-opus-20240229` | Full support with logging |
| **Google Gemini** | ✅ Active | `gemini-pro` | Full support with new API |
| **Grok (xAI)** | ✅ Active | `grok-beta` | Full support with OpenAI-compatible API |

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

### Grok (xAI)
```python
config = {
    "api_key": "your-xai-api-key",
    "model": "grok-beta",  # or other available Grok models
    "base_url": "https://api.x.ai/v1"  # Optional, defaults to xAI endpoint
}
llm = get_llm("grok", config)
```

## 📖 Usage Examples

### Basic Chat
```python
from factory import get_llm

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
providers = ["openai", "anthropic", "gemini", "grok"]
configs = {
    "openai": {"api_key": "openai_key", "model": "gpt-4"},
    "anthropic": {"api_key": "anthropic_key", "model": "claude-3-opus-20240229"},
    "gemini": {"api_key": "gemini_key", "model": "gemini-pro"},
    "grok": {"api_key": "xai_key", "model": "grok-beta"}
}

for provider in providers:
    llm = get_llm(provider, configs[provider])
    response = llm.chat([{"role": "user", "content": "Hello!"}])
    print(f"{provider}: {response}")
```

## 📊 Enhanced Logging & Monitoring

The library includes comprehensive logging for monitoring API usage with enhanced provider identification:

### Configure Logging Level
```bash
export LLMWRAPPER_LOG_LEVEL=DEBUG  # Options: DEBUG, INFO, WARNING, ERROR
```

### What Gets Logged
- **Provider Initialization**: Wrapper instantiation with provider and model details
- **API Calls**: Provider/model identification and message count
- **Response Timing**: How long each API call takes per provider
- **Token Usage**: Prompt, completion, and total tokens with provider identification
- **Errors**: Failed API calls and error details

### Enhanced Log Output
```
[2024-01-15 10:30:44] [INFO] Initialized openai wrapper with model: gpt-4
[2024-01-15 10:30:45] [INFO] Calling openai/gpt-4 with 2 message(s)
[2024-01-15 10:30:47] [INFO] openai/gpt-4 response received in 1.85 seconds
[2024-01-15 10:30:47] [INFO] openai - Prompt tokens: 45, Completion tokens: 123, Total: 168
```

### Log Files
- Logs are written to both console and `llmwrapper.log` file
- Log file creation is optional (gracefully handles permission errors)
- Provider-specific logging helps with debugging and monitoring multiple providers

## 🏗️ Architecture

### Project Structure
```
llmwrapper/
├── base.py                 # Abstract base class
├── openai_wrapper.py       # OpenAI implementation (modern SDK v1.0.0+)
├── anthropic_wrapper.py    # Anthropic Claude implementation  
├── gemini_wrapper.py       # Google Gemini implementation
├── grok_wrapper.py         # Grok placeholder implementation
├── factory.py              # Factory pattern for provider selection
├── logger.py               # Logging configuration
├── logging_mixin.py        # Logging functionality mixin
├── example_usage.py        # Usage examples with security best practices
├── tests/
│   └── test_llmwrapper.py  # Test suite
├── requirements.txt        # Dependencies
└── README.md              # This file
```

### Adding New Providers

To add a new LLM provider:

1. Create a new wrapper class extending `BaseLLM` and `LoggingMixin`:
```python
from base import BaseLLM
from logging_mixin import LoggingMixin

class NewProviderWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider = "new_provider"  # Define provider name
        self.log_provider_init(self.provider, self.model)  # Log initialization
    
    def chat(self, messages: list[dict], **kwargs) -> str:
        start = self.log_call_start(self.provider, self.model, len(messages))
        # Implement chat functionality
        response = your_api_call(messages, **kwargs)
        self.log_call_end(self.provider, self.model, start)
        
        # Log token usage if available
        usage_info = extract_usage_from_response(response)
        self.log_token_usage(self.provider, usage_info)
        
        return response.content
```

2. Update `factory.py` to include your provider:
```python
from new_provider_wrapper import NewProviderWrapper

def get_llm(provider: str, config: dict):
    # ... existing providers ...
    elif provider == "new_provider":
        return NewProviderWrapper(api_key=config["api_key"], model=config.get("model", "default-model"))
```

## 🧪 Testing

Run the comprehensive test suite:
```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_llmwrapper.py
```

### Enhanced Test Coverage
- **LoggingMixin functionality**: All logging methods with provider identification
- **Base class functionality**: Abstract base class implementation
- **Factory pattern**: Provider selection and default model handling
- **Wrapper initialization**: Provider setup and logging verification
- **Chat method testing**: API calls with comprehensive mocking
- **Error handling**: Invalid providers and missing configuration
- **Token usage logging**: Provider-specific usage tracking
- **Provider instantiation logging**: Factory-level logging verification

### Test Results
All 20 tests passing with comprehensive coverage of:
- 5 LoggingMixin tests
- 6 Factory pattern tests  
- 4 Wrapper initialization tests
- 2 Chat method tests
- 3 Error handling tests

## 🔐 API Keys Setup

You'll need API keys from the respective providers:

- **OpenAI**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic**: Get from [Anthropic Console](https://console.anthropic.com/)
- **Google Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Grok (xAI)**: Get from [xAI Console](https://console.x.ai/)

## 🌍 Environment Variables

| Variable | Purpose | Default | Example |
|----------|---------|---------|---------|
| `LLMWRAPPER_LOG_LEVEL` | Set logging level | `INFO` | `DEBUG`, `WARNING` |
| `OPENAI_API_KEY` | OpenAI authentication | - | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic authentication | - | `sk-ant-...` |
| `GEMINI_API_KEY` | Google Gemini authentication | - | `AIza...` |
| `XAI_API_KEY` | xAI Grok authentication | - | `xai-...` |

## ⚠️ Important Notes

1. **Provider-Specific Parameters**: Each provider supports unique configuration options:
   - **Grok**: Supports `base_url` parameter for custom endpoints
   - **All providers**: Support various model-specific parameters via `**kwargs`

2. **Rate Limits**: Each provider has different rate limits. Implement appropriate rate limiting in production.

3. **Error Handling**: The current implementation provides basic error handling. Consider adding more robust error handling for production use.

4. **Security**: Never commit API keys to version control. Use environment variables or secure configuration files.

5. **Logging**: Be mindful of logging sensitive information. The current implementation logs token usage and timing but not message content.

## ✅ Recent Updates

### **v2.0.0 - Enhanced Logging & Provider Management**
- **✅ Enhanced Logging System**: Added provider identification to all log messages
- **✅ Provider Initialization Logging**: Track wrapper instantiation and configuration
- **✅ Improved Token Usage Logging**: Provider-specific token usage tracking for all providers
- **✅ Google Gemini API Update**: Migrated to latest `google-genai` package
- **✅ Complete Grok Implementation**: Full xAI Grok API integration with OpenAI-compatible interface
- **✅ Provider-Specific Parameters**: Support for `base_url` and other provider-specific options
- **✅ Comprehensive Test Suite**: 20 tests with advanced mocking and complete coverage
- **✅ Better Error Handling**: Enhanced error reporting and logging consistency

### **v1.1.0 - Critical Fixes**
- **✅ OpenAI API Modernization**: Updated to use OpenAI SDK v1.0.0+ client-based approach
- **✅ Anthropic Logging Fix**: Resolved duplicate logging calls causing incorrect timing measurements  
- **✅ Enhanced Testing**: Updated test suite with proper mocking for new API structures
- **✅ Improved Requirements**: Added pytest-mock for better testing capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Run the test suite: `pytest`
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.
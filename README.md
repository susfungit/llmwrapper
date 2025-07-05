# LLM Wrapper

A vendor-agnostic Python wrapper for interacting with multiple Large Language Models (LLMs) including OpenAI, Anthropic Claude, Google Gemini, Grok (xAI), and local models via Ollama.

## 🚀 Features

- **Unified Interface**: Single API to interact with multiple LLM providers (cloud + local)
- **Registry Pattern**: Modern decorator-based provider registration system
- **Easy Provider Switching**: Change providers with minimal code changes
- **🦙 Local LLM Support**: Run models locally with Ollama (Llama, Mistral, CodeLlama, etc.)
- **⚡ Full Async Support**: High-performance concurrent operations with asyncio
- **🚀 Concurrent Requests**: Make multiple API calls simultaneously for better performance
- **🔒 Privacy-First**: Keep sensitive data local with on-premise model inference
- **Extensible**: Easy to add new providers with decorators - no factory modifications needed
- **Type Safety**: Full type hints for better development experience
- **🔒 Enterprise Security**: Comprehensive security features with automatic credential masking and input validation
- **🛡️ Secure Logging**: Automatic API key masking and sensitive data protection in logs
- **✅ Input Validation**: Built-in validation for API keys, messages, and parameters with injection attack prevention
- **🔐 Security Event Logging**: Automatic logging of security events with sanitized details
- **Enhanced Logging**: Comprehensive logging with provider/model identification, timing, and token usage
- **Provider Initialization Logging**: Track wrapper instantiation and configuration
- **Comprehensive Testing**: Full test coverage with pytest and advanced mocking (91 tests: 89 passed, 2 skipped)
- **Secure Configuration**: Environment variable and config file support
- **Modern API Support**: Uses latest OpenAI SDK v1.0.0+ and Google Gemini API

## 🔧 Installation

### Python Package
```bash
pip install llmwrapper
```

### For Local LLM Support (Ollama)
```bash
# 1. Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 2. Start Ollama server
ollama serve

# 3. Pull a model (in a new terminal)
ollama pull llama3
```

For development installation:

```bash
git clone https://github.com/yourusername/llmwrapper.git
cd llmwrapper
pip install -e .
```

## 🚀 Quick Start

```python
from llmwrapper import get_llm

# OpenAI Example (with automatic security features)
config = {
    "api_key": "your-openai-api-key",  # Automatically masked in logs as "***"
    "model": "gpt-4"
}

llm = get_llm("openai", config)  # Input validation happens automatically
response = llm.chat([
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's the capital of France?"}
])
print(response)

# 🔒 Security features work automatically:
# ✅ API keys are masked in all log outputs
# ✅ Input validation prevents injection attacks
# ✅ Security events are logged with sanitized details
# ✅ No additional configuration required!
```

## 📦 Supported Providers

| Provider | Type | Status | Default Model | Notes |
|----------|------|--------|---------------|-------|
| **OpenAI** | Cloud | ✅ Active | `gpt-4` | Full support with logging, modern SDK v1.0.0+ |
| **Anthropic** | Cloud | ✅ Active | `claude-3-opus-20240229` | Full support with logging |
| **Google Gemini** | Cloud | ✅ Active | `gemini-pro` | Full support with new API |
| **Grok (xAI)** | Cloud | ✅ Active | `grok-beta` | Full support with OpenAI-compatible API |
| **Ollama** | Local | ✅ Active | `llama3` | Local inference, supports Llama, Mistral, CodeLlama+ |

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

### Ollama (Local LLMs)
```python
config = {
    "model": "llama3",  # or "mistral", "codellama", "phi", etc.
    "base_url": "http://localhost:11434",  # Optional, defaults to localhost
    "api_key": None  # Not required for local inference
}
llm = get_llm("ollama", config)

# Available models (pull with: ollama pull <model>)
# - llama3, llama3:70b
# - mistral, mistral:7b
# - codellama, codellama:34b
# - phi, phi3
# - And many more at https://ollama.ai/library
```

## 📖 Usage Examples

### Basic Chat
```python
from llmwrapper import get_llm

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
# Easy to switch between cloud and local providers
providers = ["openai", "anthropic", "gemini", "grok", "ollama"]
configs = {
    "openai": {"api_key": "openai_key", "model": "gpt-4"},
    "anthropic": {"api_key": "anthropic_key", "model": "claude-3-opus-20240229"},
    "gemini": {"api_key": "gemini_key", "model": "gemini-pro"},
    "grok": {"api_key": "xai_key", "model": "grok-beta"},
    "ollama": {"api_key": None, "model": "llama3", "base_url": "http://localhost:11434"}
}

for provider in providers:
    llm = get_llm(provider, configs[provider])
    response = llm.chat([{"role": "user", "content": "Hello!"}])
    print(f"{provider}: {response}")
```

### Hybrid Usage (Cloud + Local)
```python
# Use local for sensitive data, cloud for complex reasoning
sensitive_llm = get_llm("ollama", {"model": "llama3", "api_key": None})
complex_llm = get_llm("openai", {"api_key": "your-key", "model": "gpt-4"})

# Process sensitive data locally
sensitive_response = sensitive_llm.chat([
    {"role": "user", "content": "Analyze this confidential document..."}
])

# Use cloud for complex reasoning
complex_response = complex_llm.chat([
    {"role": "user", "content": "Solve this complex mathematical proof..."}
])
```

## ⚡ Async Support

The library includes full async support for high-performance concurrent operations:

### Basic Async Usage
```python
import asyncio
from llmwrapper import get_async_llm

async def main():
    # Create async LLM instance
    llm = get_async_llm("openai", {
        "api_key": "your-openai-api-key",
        "model": "gpt-4"
    })
    
    # Make async request
    response = await llm.chat([
        {"role": "user", "content": "What is async programming?"}
    ])
    print(response)

# Run async function
asyncio.run(main())
```

### Concurrent Requests to Multiple Providers
```python
import asyncio
from llmwrapper import get_async_llm

async def concurrent_requests():
    # Create multiple async LLM instances
    openai_llm = get_async_llm("openai", {"api_key": "key1", "model": "gpt-4"})
    anthropic_llm = get_async_llm("anthropic", {"api_key": "key2", "model": "claude-3-opus-20240229"})
    
    # Make concurrent requests
    question = [{"role": "user", "content": "What is 2+2?"}]
    
    # Both requests happen simultaneously
    openai_response, anthropic_response = await asyncio.gather(
        openai_llm.chat(question),
        anthropic_llm.chat(question)
    )
    
    print(f"OpenAI: {openai_response}")
    print(f"Anthropic: {anthropic_response}")

asyncio.run(concurrent_requests())
```

### Batch Processing Multiple Questions
```python
import asyncio
from llmwrapper import get_async_llm

async def batch_processing():
    llm = get_async_llm("openai", {"api_key": "your-key", "model": "gpt-4"})
    
    questions = [
        "What is the capital of France?",
        "What is 10 + 15?",
        "Name a programming language.",
        "What color is the sky?",
    ]
    
    # Process all questions concurrently
    tasks = []
    for question in questions:
        task = llm.chat([{"role": "user", "content": question}])
        tasks.append(task)
    
    # Wait for all responses
    responses = await asyncio.gather(*tasks)
    
    for question, response in zip(questions, responses):
        print(f"Q: {question}")
        print(f"A: {response}\n")

asyncio.run(batch_processing())
```

### Performance Benefits

Async operations provide significant performance improvements for I/O-bound tasks:

- **Concurrent Requests**: Make multiple API calls simultaneously instead of sequentially
- **Better Resource Utilization**: Don't block while waiting for API responses
- **Scalability**: Handle many requests with minimal resource overhead

**Example Performance Comparison:**
```
Sequential (sync): 3 requests × 2 seconds each = 6 seconds total
Concurrent (async): 3 requests simultaneously = ~2 seconds total
```

### Async Provider Support

| Provider | Async Support | Implementation |
|----------|---------------|----------------|
| **OpenAI** | ✅ Full | `AsyncOpenAI` client |
| **Anthropic** | ✅ Full | `AsyncAnthropic` client |
| **Gemini** | ✅ Partial | Thread pool executor* |
| **Grok** | ✅ Full | `AsyncOpenAI` client |
| **Ollama** | ✅ Full | `aiohttp` client |

*Gemini uses thread pool execution since the google-genai library doesn't have native async support yet.

### Running Async Examples
```bash
python async_example_usage.py
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
- **🔒 Security Events**: Automatic logging of security-related events with masked sensitive data
- **🛡️ Credential Protection**: All API keys and sensitive data automatically masked

### Enhanced Log Output (with Security)
```
[2024-01-15 10:30:44] [INFO] Initialized openai wrapper with model: gpt-4
[2024-01-15 10:30:45] [INFO] Calling openai/gpt-4 with 2 message(s)
[2024-01-15 10:30:47] [INFO] openai/gpt-4 response received in 1.85 seconds
[2024-01-15 10:30:47] [INFO] openai - Prompt tokens: 45, Completion tokens: 123, Total: 168
[2024-01-15 10:30:48] [WARNING] SECURITY EVENT - INVALID_API_KEY: {"provider": "openai", "api_key_format": "invalid"}
```

### Log Files
- Logs are written to both console and `llmwrapper.log` file
- Log file creation is optional (gracefully handles permission errors)
- Provider-specific logging helps with debugging and monitoring multiple providers

## 🔒 Enterprise Security Features

The LLM wrapper includes comprehensive security features to protect sensitive information and prevent security vulnerabilities:

### 🛡️ Automatic Credential Masking

All API keys, tokens, and sensitive data are automatically masked in logs:

```python
from llmwrapper import get_llm

# API keys are automatically masked in logs
llm = get_llm("openai", {
    "api_key": "sk-1234567890abcdef1234567890abcdef",
    "model": "gpt-4"
})

# Log output: "sk-***" instead of full API key
```

**Supported Masking Patterns:**
- **OpenAI keys**: `sk-***` (from `sk-1234567890abcdef...`)
- **Anthropic keys**: `sk-ant-***` (from `sk-ant-api03-abcdef...`)
- **Google Gemini keys**: `AIza***` (from `AIzaSyDabcdef...`)
- **Grok (xAI) keys**: `xai-***` (from `xai-abcdef...`)
- **Bearer tokens**: `Bearer ***`
- **URL credentials**: `https://user:***@domain.com`
- **Generic secrets**: Long alphanumeric strings are automatically detected and masked

### ✅ Input Validation

Built-in validation protects against common security vulnerabilities:

```python
from llmwrapper.security_utils import SecurityUtils

# API key validation per provider
is_valid = SecurityUtils.validate_api_key("sk-1234567890abcdef...", "openai")

# Message validation with injection detection
messages = [{"role": "user", "content": "Hello world"}]
is_safe = SecurityUtils.validate_messages(messages)

# URL validation
is_valid_url = SecurityUtils.validate_url("https://api.openai.com/v1")
```

**Validation Features:**
- **API Key Format Validation**: Provider-specific format checking
- **Message Structure Validation**: Ensures proper role/content format
- **Injection Attack Prevention**: Detects XSS, eval, exec, and system command attempts
- **URL Validation**: Ensures only HTTP/HTTPS schemes are used
- **Parameter Validation**: Temperature (0-2), max_tokens (1-32768), etc.

### 🔐 Security Event Logging

Automatic logging of security-related events with sanitized details:

```python
# Security events are automatically logged when:
# - Invalid API keys are detected
# - Injection attempts are blocked
# - Unexpected errors occur
# - Parameter validation fails

# Example log output:
# [2024-01-15 10:30:45] [WARNING] SECURITY EVENT - INVALID_API_KEY: {"provider": "openai", "api_key_format": "invalid"}
# [2024-01-15 10:30:46] [WARNING] SECURITY EVENT - INJECTION_ATTEMPT: {"provider": "openai", "pattern": "script_tag"}
```

### 🛡️ Secure Data Handling

Comprehensive data sanitization for complex structures:

```python
from llmwrapper.security_utils import SecurityUtils

# Recursive masking of nested data structures
sensitive_data = {
    "config": {
        "api_key": "sk-1234567890abcdef1234567890abcdef",
        "model": "gpt-4"
    },
    "credentials": {
        "secret": "topsecret",
        "password": "secret123"
    }
}

# Automatically masks sensitive fields while preserving safe data
masked_data = SecurityUtils.mask_sensitive_data(sensitive_data)
# Result: {"config": {"api_key": "sk-***", "model": "gpt-4"}, "credentials": {"secret": "***", "password": "***"}}
```

### 🔍 Security Best Practices

**Environment Variables (Recommended):**
```bash
# Use environment variables for API keys
export OPENAI_API_KEY="sk-your-real-key-here"
export ANTHROPIC_API_KEY="sk-ant-your-real-key-here"
export GEMINI_API_KEY="AIza-your-real-key-here"
export XAI_API_KEY="xai-your-real-key-here"
```

```python
import os
from llmwrapper import get_llm

# Secure usage with environment variables
llm = get_llm("openai", {
    "api_key": os.getenv("OPENAI_API_KEY"),
    "model": "gpt-4"
})
```

**Security Configuration:**
```python
# Enable secure logging (default)
export LLMWRAPPER_LOG_LEVEL=INFO

# The library automatically:
# ✅ Masks all API keys in logs
# ✅ Validates input parameters
# ✅ Prevents injection attacks
# ✅ Logs security events
# ✅ Sanitizes error messages
```

### 🔒 Security Testing

Comprehensive security test suite validates all security features:

```bash
# Run security-specific tests
pytest tests/test_security.py -v
pytest tests/test_security_logging.py -v

# Security test coverage:
# - API key validation (16 tests)
# - Data masking (4 tests)
# - Secure logging (6 tests)
# - Input validation (8 tests)
# - Security event logging (3 tests)
# - Integration tests (3 tests)
```

### ⚠️ Security Considerations

1. **API Key Management**: Never hardcode API keys in source code
2. **Log File Security**: Ensure log files are properly secured with appropriate file permissions
3. **Network Security**: Use HTTPS endpoints only (validated automatically)
4. **Input Sanitization**: The library validates inputs, but review application-specific data
5. **Error Handling**: Security events are logged but don't interrupt normal operation

### 🔐 Security Documentation

For detailed security information, see:
- `SECURITY_RECOMMENDATIONS.md` - Comprehensive security guidelines
- `llmwrapper/security_utils.py` - Security utilities implementation
- `tests/test_security*.py` - Security test suite

## ��️ Architecture

### Registry Pattern with Decorators
The library uses a modern registry pattern with decorators for clean, extensible provider management:

```python
# Providers self-register using decorators
@register_sync_provider("openai", "gpt-4")
class OpenAIWrapper(BaseLLM, LoggingMixin):
    # Implementation...

@register_async_provider("anthropic", "claude-3-opus-20240229")
class AsyncClaudeWrapper(AsyncBaseLLM, LoggingMixin):
    # Implementation...
```

### Project Structure
```
llmwrapper/
├── base.py                      # Abstract base class
├── async_base.py                # Abstract async base class
├── registry.py                  # Registry system with decorators
├── security_utils.py            # Security utilities (credential masking, validation, logging)
├── openai_wrapper.py            # OpenAI sync implementation (modern SDK v1.0.0+)
├── async_openai_wrapper.py      # OpenAI async implementation
├── anthropic_wrapper.py         # Anthropic Claude sync implementation  
├── async_anthropic_wrapper.py   # Anthropic Claude async implementation
├── gemini_wrapper.py            # Google Gemini sync implementation
├── async_gemini_wrapper.py      # Google Gemini async implementation
├── grok_wrapper.py              # Grok sync implementation
├── async_grok_wrapper.py        # Grok async implementation
├── ollama_wrapper.py            # Ollama local LLM sync implementation
├── async_ollama_wrapper.py      # Ollama local LLM async implementation
├── factory.py                   # Factory pattern for sync provider selection
├── async_factory.py             # Factory pattern for async provider selection
├── logger.py                    # Secure logging configuration with automatic masking
├── logging_mixin.py             # Logging functionality mixin
├── examples/
│   ├── example_usage.py         # Sync usage examples with security best practices
│   ├── async_example_usage.py   # Async usage examples and performance demos
│   ├── ollama_example_usage.py  # Ollama local LLM sync examples
│   └── async_ollama_example_usage.py # Ollama local LLM async examples
├── tests/
│   ├── test_llmwrapper.py       # Sync test suite (with security features)
│   ├── test_async_llmwrapper.py # Async test suite (with security features)
│   ├── test_security.py         # Security utilities test suite
│   └── test_security_logging.py # Secure logging test suite
├── requirements.txt             # Dependencies (including security packages)
├── SECURITY_RECOMMENDATIONS.md # Comprehensive security guidelines
└── README.md                   # This file
```

### Adding New Providers

To add a new LLM provider using the registry pattern:

1. Create a new wrapper class with the decorator:
```python
from .base import BaseLLM
from .logging_mixin import LoggingMixin
from .registry import register_sync_provider

@register_sync_provider("new_provider", "default-model")
class NewProviderWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.provider = "new_provider"
        self.log_provider_init(self.provider, self.model)
    
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

2. For async providers, use the async decorator:
```python
from .async_base import AsyncBaseLLM
from .registry import register_async_provider

@register_async_provider("new_provider", "default-model")
class AsyncNewProviderWrapper(AsyncBaseLLM, LoggingMixin):
    # Async implementation...
```

3. **No factory.py changes needed!** The registry pattern automatically handles provider registration.

### Benefits of Registry Pattern
- **Extensible**: Add providers without modifying factory code
- **Clean**: No if/elif chains in factory functions
- **Automatic**: Providers self-register when imported
- **Type-Safe**: Decorator ensures proper provider configuration

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
- **Factory pattern**: Provider selection and default model handling with registry pattern
- **Wrapper initialization**: Provider setup and logging verification
- **Chat method testing**: API calls with comprehensive mocking
- **Error handling**: Invalid providers and missing configuration
- **Token usage logging**: Provider-specific usage tracking
- **Provider instantiation logging**: Factory-level logging verification
- **Async functionality**: Complete async test suite for all providers
- **Concurrent operations**: Async batch processing and error handling
- **Security features**: Comprehensive security validation and logging tests
- **Input validation**: API key, message, URL, and parameter validation
- **Data masking**: Recursive masking of sensitive data in complex structures
- **Secure logging**: Automatic credential masking in log outputs

### Test Results
All 91 tests (89 passed, 2 skipped) with comprehensive coverage of:
- **32 Sync Tests**: LoggingMixin (5), Factory (7), Wrapper Init (6), Chat Methods (5), Error Handling (3), Security Features (6)
- **20 Async Tests**: Async factory (6), Async wrappers (10), Concurrent operations (1), Async error handling (1), Async security (2)
- **29 Security Tests**: SecurityUtils (16), SecureLogging (4), WrapperSecurity (6), SecurityIntegration (3)
- **10 Security Logging Tests**: End-to-end secure logging validation and integration tests
- **2 Skipped Tests**: Complex async HTTP mocking tests (validated via integration testing)

## 🔐 API Keys Setup

You'll need API keys from the respective cloud providers:

- **OpenAI**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Anthropic**: Get from [Anthropic Console](https://console.anthropic.com/)
- **Google Gemini**: Get from [Google AI Studio](https://makersuite.google.com/app/apikey)
- **Grok (xAI)**: Get from [xAI Console](https://console.x.ai/)
- **Ollama**: No API key required - runs locally! 🎉

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

5. **🔒 Security Features**: The library automatically protects sensitive information with built-in security features:
   - API keys are automatically masked in all logs
   - Input validation prevents injection attacks
   - Security events are logged with sanitized details
   - No additional configuration required for basic security

6. **Logging**: All sensitive information is automatically masked. The implementation logs token usage and timing but not message content.

## ✅ Recent Updates

### **v1.0.0 - Registry Pattern & Local LLM Support**
- **✅ Registry Pattern Implementation**: Modern decorator-based provider registration system
- **✅ Ollama Integration**: Local LLM support (Llama, Mistral, CodeLlama, etc.)
- **✅ Eliminated If/Elif Chains**: Clean, extensible provider management
- **✅ 46 Comprehensive Tests**: Full test suite covering sync/async functionality
- **✅ Enhanced Logging System**: Added provider identification to all log messages
- **✅ Provider Initialization Logging**: Track wrapper instantiation and configuration
- **✅ Privacy-First Option**: Keep sensitive data local with on-premise inference
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

### **v1.2.0 - Enterprise Security**
- **🔒 Comprehensive Security Implementation**: Enterprise-grade security features with automatic credential protection
- **🛡️ Automatic Credential Masking**: All API keys, tokens, and sensitive data automatically masked in logs
- **✅ Input Validation System**: Built-in validation for API keys, messages, URLs, and parameters
- **🔐 Security Event Logging**: Automatic logging of security events with sanitized details
- **🚫 Injection Attack Prevention**: Detection and blocking of XSS, eval, exec, and system command attempts
- **📊 Recursive Data Masking**: Secure handling of complex nested data structures
- **🔍 Provider-Specific Validation**: Custom validation rules for each LLM provider
- **🧪 Comprehensive Security Testing**: 40+ security tests covering all security features
- **📋 Security Documentation**: Detailed security guidelines and best practices
- **⚡ Zero Performance Impact**: Security features designed for production use without performance degradation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes with tests
4. Run the test suite: `pytest`
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.
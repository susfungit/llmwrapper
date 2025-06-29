# llmwrapper

A vendor-agnostic Python wrapper for interacting with multiple Large Language Models (LLMs) like OpenAI and Anthropic.

## 🔧 Installation
```bash
pip install -r requirements.txt
```

## 🚀 Usage
```python
from llmwrapper.factory import get_llm

config = {
    "api_key": "your-api-key",
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
- OpenAI
- Anthropic (Claude)

## 📁 Structure
```
llmwrapper/
├── base.py
├── openai_wrapper.py
├── anthropic_wrapper.py
├── factory.py
example_usage.py
requirements.txt
.gitignore
README.md
```
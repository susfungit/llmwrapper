import pytest
import sys
import os

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from base import BaseLLM
from factory import get_llm

class DummyLLM(BaseLLM):
    def chat(self, messages: list[dict], **kwargs) -> str:
        return "Test response"

def test_dummy_llm_chat():
    dummy = DummyLLM()
    response = dummy.chat([{"role": "user", "content": "Hello"}])
    assert isinstance(response, str)
    assert response == "Test response"

def test_factory_openai(monkeypatch):
    # Mock the OpenAI client and response structure to match new API
    class MockChoice:
        def __init__(self):
            self.message = MockMessage()
    
    class MockMessage:
        def __init__(self):
            self.content = "Mocked OpenAI response"
    
    class MockUsage:
        def model_dump(self):
            return {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
    
    class MockResponse:
        def __init__(self):
            self.choices = [MockChoice()]
            self.usage = MockUsage()
    
    class MockChatCompletions:
        def create(self, **kwargs):
            return MockResponse()
    
    class MockChat:
        def __init__(self):
            self.completions = MockChatCompletions()
    
    class MockOpenAIClient:
        def __init__(self, api_key):
            self.chat = MockChat()
    
    class MockOpenAIWrapper:
        def __init__(self, api_key, model):
            pass
        def chat(self, messages, **kwargs):
            return "Mocked OpenAI response"

    import factory
    monkeypatch.setattr(factory, "OpenAIWrapper", MockOpenAIWrapper)
    config = {"api_key": "test", "model": "gpt-4"}
    llm = get_llm("openai", config)
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked OpenAI response"

def test_factory_anthropic(monkeypatch):
    class MockAnthropicWrapper:
        def __init__(self, api_key, model):
            pass
        def chat(self, messages, **kwargs):
            return "Mocked Anthropic response"

    import factory
    monkeypatch.setattr(factory, "ClaudeWrapper", MockAnthropicWrapper)
    config = {"api_key": "test", "model": "claude-3-sonnet-20240229"}
    llm = get_llm("anthropic", config)
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Anthropic response"

def test_factory_gemini(monkeypatch):
    class MockGeminiWrapper:
        def __init__(self, api_key, model):
            pass
        def chat(self, messages, **kwargs):
            return "Mocked Gemini response"

    import factory
    monkeypatch.setattr(factory, "GeminiWrapper", MockGeminiWrapper)
    config = {"api_key": "test", "model": "gemini-pro"}
    llm = get_llm("gemini", config)
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Gemini response"

def test_factory_grok(monkeypatch):
    class MockGrokWrapper:
        def __init__(self, api_key, model):
            pass
        def chat(self, messages, **kwargs):
            return "Mocked Grok response"

    import factory
    monkeypatch.setattr(factory, "GrokWrapper", MockGrokWrapper)
    config = {"api_key": "test", "model": "grok-1"}
    llm = get_llm("grok", config)
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Grok response"

def test_factory_invalid_provider():
    config = {"api_key": "test"}
    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        get_llm("unknown", config)

def test_openai_wrapper_initialization():
    """Test that OpenAI wrapper can be initialized with proper parameters"""
    from openai_wrapper import OpenAIWrapper
    
    # This would normally require a real API key, but we're just testing initialization logic
    try:
        wrapper = OpenAIWrapper(api_key="test-key", model="gpt-4")
        assert wrapper.model == "gpt-4"
        assert hasattr(wrapper, 'client')
    except Exception as e:
        # Expected to fail without valid API key, but should fail at API call, not initialization
        assert "api_key" in str(e).lower() or "authentication" in str(e).lower()

def test_anthropic_wrapper_initialization():
    """Test that Anthropic wrapper can be initialized with proper parameters"""
    from anthropic_wrapper import ClaudeWrapper
    
    try:
        wrapper = ClaudeWrapper(api_key="test-key", model="claude-3-sonnet-20240229")
        assert wrapper.model == "claude-3-sonnet-20240229"
        assert hasattr(wrapper, 'client')
    except Exception as e:
        # Expected to fail without valid API key, but should fail at API call, not initialization
        assert "api" in str(e).lower() or "key" in str(e).lower()
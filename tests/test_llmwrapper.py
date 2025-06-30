import pytest
from llmwrapper.base import BaseLLM
from llmwrapper.factory import get_llm

class DummyLLM(BaseLLM):
    def chat(self, messages: list[dict], **kwargs) -> str:
        return "Test response"

def test_dummy_llm_chat():
    dummy = DummyLLM()
    response = dummy.chat([{"role": "user", "content": "Hello"}])
    assert isinstance(response, str)
    assert response == "Test response"

def test_factory_openai(monkeypatch):
    class MockOpenAI:
        def __init__(self, api_key, model):
            pass
        def chat(self, messages, **kwargs):
            return "Mocked OpenAI response"

    from llmwrapper import factory
    monkeypatch.setattr(factory, "OpenAIWrapper", MockOpenAI)
    config = {"api_key": "test", "model": "gpt-4"}
    llm = get_llm("openai", config)
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked OpenAI response"

def test_factory_gemini(monkeypatch):
    class MockGemini:
        def __init__(self, api_key, model):
            pass
        def chat(self, messages, **kwargs):
            return "Mocked Gemini response"

    from llmwrapper import factory
    monkeypatch.setattr(factory, "GeminiWrapper", MockGemini)
    config = {"api_key": "test", "model": "gemini-pro"}
    llm = get_llm("gemini", config)
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Gemini response"

def test_factory_grok(monkeypatch):
    class MockGrok:
        def __init__(self, api_key, model):
            pass
        def chat(self, messages, **kwargs):
            return "Mocked Grok response"

    from llmwrapper import factory
    monkeypatch.setattr(factory, "GrokWrapper", MockGrok)
    config = {"api_key": "test", "model": "grok-1"}
    llm = get_llm("grok", config)
    assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Grok response"

def test_factory_invalid_provider():
    config = {"api_key": "test"}
    with pytest.raises(ValueError):
        get_llm("unknown", config)
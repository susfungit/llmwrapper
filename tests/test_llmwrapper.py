import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
import time

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llmwrapper import BaseLLM, get_llm, LoggingMixin

class DummyLLM(BaseLLM):
    def chat(self, messages: list[dict], **kwargs) -> str:
        return "Test response"

class TestLoggingMixin:
    """Test the LoggingMixin functionality"""
    
    def setup_method(self):
        self.mixin = LoggingMixin()
    
    @patch('llmwrapper.logging_mixin.logger')
    def test_log_call_start(self, mock_logger):
        start_time = self.mixin.log_call_start("openai", "gpt-4", 2)
        
        mock_logger.info.assert_called_once_with("Calling openai/gpt-4 with 2 message(s)")
        assert isinstance(start_time, float)
        assert start_time <= time.time()
    
    @patch('llmwrapper.logging_mixin.logger')
    def test_log_call_end(self, mock_logger):
        start_time = time.time() - 1.5  # Simulate 1.5 seconds ago
        self.mixin.log_call_end("anthropic", "claude-3-sonnet", start_time)
        
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "anthropic/claude-3-sonnet response received in" in call_args
        assert "seconds" in call_args
    
    @patch('llmwrapper.logging_mixin.logger')
    def test_log_token_usage_with_usage(self, mock_logger):
        usage = {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30
        }
        self.mixin.log_token_usage("openai", usage)
        
        mock_logger.info.assert_called_once_with(
            "openai - Prompt tokens: 10, Completion tokens: 20, Total: 30"
        )
    
    @patch('llmwrapper.logging_mixin.logger')
    def test_log_token_usage_without_usage(self, mock_logger):
        self.mixin.log_token_usage("gemini", {})
        
        mock_logger.warning.assert_called_once_with(
            "gemini - Token usage information not available."
        )
    
    @patch('llmwrapper.logging_mixin.logger')
    def test_log_provider_init(self, mock_logger):
        self.mixin.log_provider_init("openai", "gpt-4")
        
        mock_logger.info.assert_called_once_with(
            "Initialized openai wrapper with model: gpt-4"
        )

class TestBaseLLM:
    """Test the abstract base class"""
    
    def test_dummy_llm_chat(self):
        dummy = DummyLLM()
        response = dummy.chat([{"role": "user", "content": "Hello"}])
        assert isinstance(response, str)
        assert response == "Test response"

class TestFactory:
    """Test the factory pattern implementation"""
    
    def test_factory_openai(self):
        """Test factory with OpenAI provider using registry mocking"""
        class MockOpenAIWrapper:
            def __init__(self, api_key, model):
                self.api_key = api_key
                self.model = model
                self.provider = "openai"
            def chat(self, messages, **kwargs):
                return "Mocked OpenAI response"
        
        # Mock the registry to return a fake provider info
        mock_provider_info = {
            'class': MockOpenAIWrapper,
            'default_model': 'gpt-4',
            'config': {}
        }
        
        with patch('llmwrapper.registry.llm_registry.get_sync_provider') as mock_get_provider:
            mock_get_provider.return_value = mock_provider_info
            
            config = {"api_key": "test", "model": "gpt-4"}
            llm = get_llm("openai", config)
            assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked OpenAI response"
            assert llm.model == "gpt-4"
            assert llm.provider == "openai"

    def test_factory_anthropic(self):
        """Test factory with Anthropic provider using registry mocking"""
        class MockAnthropicWrapper:
            def __init__(self, api_key, model):
                self.api_key = api_key
                self.model = model
                self.provider = "anthropic"
            def chat(self, messages, **kwargs):
                return "Mocked Anthropic response"
        
        mock_provider_info = {
            'class': MockAnthropicWrapper,
            'default_model': 'claude-3-opus-20240229',
            'config': {}
        }
        
        with patch('llmwrapper.registry.llm_registry.get_sync_provider') as mock_get_provider:
            mock_get_provider.return_value = mock_provider_info
            
            config = {"api_key": "test", "model": "claude-3-sonnet-20240229"}
            llm = get_llm("anthropic", config)
            assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Anthropic response"
            assert llm.model == "claude-3-sonnet-20240229"
            assert llm.provider == "anthropic"

    def test_factory_gemini(self):
        """Test factory with Gemini provider using registry mocking"""
        class MockGeminiWrapper:
            def __init__(self, api_key, model):
                self.api_key = api_key
                self.model = model
                self.provider = "gemini"
            def chat(self, messages, **kwargs):
                return "Mocked Gemini response"
        
        mock_provider_info = {
            'class': MockGeminiWrapper,
            'default_model': 'gemini-pro',
            'config': {}
        }
        
        with patch('llmwrapper.registry.llm_registry.get_sync_provider') as mock_get_provider:
            mock_get_provider.return_value = mock_provider_info
            
            config = {"api_key": "test", "model": "gemini-pro"}
            llm = get_llm("gemini", config)
            assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Gemini response"
            assert llm.model == "gemini-pro"
            assert llm.provider == "gemini"

    def test_factory_grok(self):
        """Test factory with Grok provider using registry mocking"""
        class MockGrokWrapper:
            def __init__(self, api_key, model, base_url):
                self.api_key = api_key
                self.model = model
                self.base_url = base_url
                self.provider = "grok"
            def chat(self, messages, **kwargs):
                return "Mocked Grok response"
        
        mock_provider_info = {
            'class': MockGrokWrapper,
            'default_model': 'grok-beta',
            'config': {'base_url': 'https://api.x.ai/v1'}
        }
        
        with patch('llmwrapper.registry.llm_registry.get_sync_provider') as mock_get_provider:
            mock_get_provider.return_value = mock_provider_info
            
            config = {"api_key": "test", "model": "grok-beta", "base_url": "https://api.x.ai/v1"}
            llm = get_llm("grok", config)
            assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Grok response"
            assert llm.model == "grok-beta"
            assert llm.provider == "grok"
            assert llm.base_url == "https://api.x.ai/v1"

    def test_factory_ollama(self):
        """Test factory with Ollama provider using registry mocking"""
        class MockOllamaWrapper:
            def __init__(self, api_key, model, base_url):
                self.api_key = api_key
                self.model = model
                self.base_url = base_url
                self.provider = "ollama"
            def chat(self, messages, **kwargs):
                return "Mocked Ollama response"
            def list_models(self):
                return ["llama3", "mistral", "codellama"]
        
        mock_provider_info = {
            'class': MockOllamaWrapper,
            'default_model': 'llama3',
            'config': {'base_url': 'http://localhost:11434'}
        }
        
        with patch('llmwrapper.registry.llm_registry.get_sync_provider') as mock_get_provider:
            mock_get_provider.return_value = mock_provider_info
            
            config = {"api_key": None, "model": "llama3", "base_url": "http://localhost:11434"}
            llm = get_llm("ollama", config)
            assert llm.chat([{"role": "user", "content": "Hi"}]) == "Mocked Ollama response"
            assert llm.model == "llama3"
            assert llm.provider == "ollama"
            assert llm.base_url == "http://localhost:11434"
            assert llm.list_models() == ["llama3", "mistral", "codellama"]

    def test_factory_invalid_provider(self):
        config = {"api_key": "test"}
        with pytest.raises(ValueError, match="Unsupported sync provider"):
            get_llm("unknown", config)

    def test_factory_default_models(self, monkeypatch):
        """Test that default models are used when not specified"""
        class MockWrapper:
            def __init__(self, api_key, model):
                self.model = model
        
        import llmwrapper.factory
        monkeypatch.setattr(llmwrapper.factory, "OpenAIWrapper", MockWrapper)
        
        config = {"api_key": "test"}  # No model specified
        llm = get_llm("openai", config)
        assert llm.model == "gpt-4"  # Should use default

class TestWrapperInitialization:
    """Test wrapper initialization and provider attributes"""

    def test_openai_wrapper_initialization(self):
        """Test that OpenAI wrapper can be initialized with proper parameters"""
        from llmwrapper.openai_wrapper import OpenAIWrapper
        
        # Mock the OpenAI client to avoid actual API calls
        with patch('llmwrapper.openai_wrapper.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch.object(LoggingMixin, 'log_provider_init') as mock_log:
                wrapper = OpenAIWrapper(api_key="test-key", model="gpt-4")
                
                assert wrapper.model == "gpt-4"
                assert wrapper.provider == "openai"
                assert hasattr(wrapper, 'client')
                mock_log.assert_called_once_with("openai", "gpt-4")

    def test_anthropic_wrapper_initialization(self):
        """Test that Anthropic wrapper can be initialized with proper parameters"""
        from llmwrapper.anthropic_wrapper import ClaudeWrapper
        
        with patch('llmwrapper.anthropic_wrapper.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            with patch.object(LoggingMixin, 'log_provider_init') as mock_log:
                wrapper = ClaudeWrapper(api_key="test-key", model="claude-3-sonnet-20240229")
                
                assert wrapper.model == "claude-3-sonnet-20240229"
                assert wrapper.provider == "anthropic"
                assert hasattr(wrapper, 'client')
                mock_log.assert_called_once_with("anthropic", "claude-3-sonnet-20240229")

    def test_gemini_wrapper_initialization(self):
        """Test that Gemini wrapper can be initialized with proper parameters"""
        from llmwrapper.gemini_wrapper import GeminiWrapper
        
        with patch('llmwrapper.gemini_wrapper.genai.Client') as mock_genai:
            mock_client = Mock()
            mock_genai.return_value = mock_client
            
            with patch.object(LoggingMixin, 'log_provider_init') as mock_log:
                wrapper = GeminiWrapper(api_key="test-key", model="gemini-pro")
                
                assert wrapper.model == "gemini-pro"
                assert wrapper.provider == "gemini"
                assert hasattr(wrapper, 'client')
                mock_log.assert_called_once_with("gemini", "gemini-pro")

    def test_grok_wrapper_initialization(self):
        """Test that Grok wrapper can be initialized with proper parameters"""
        from llmwrapper.grok_wrapper import GrokWrapper
        
        with patch('llmwrapper.grok_wrapper.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_openai.return_value = mock_client
            
            with patch.object(LoggingMixin, 'log_provider_init') as mock_log:
                wrapper = GrokWrapper(api_key="test-key", model="grok-beta")
                
                assert wrapper.model == "grok-beta"
                assert wrapper.provider == "grok"
                assert wrapper.api_key == "test-key"
                assert wrapper.base_url == "https://api.x.ai/v1"
                mock_log.assert_called_once_with("grok", "grok-beta")
                
                # Verify OpenAI client was initialized with correct parameters
                mock_openai.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://api.x.ai/v1"
                )

    def test_ollama_wrapper_initialization(self):
        """Test that Ollama wrapper can be initialized with proper parameters"""
        from llmwrapper.ollama_wrapper import OllamaWrapper
        
        with patch('llmwrapper.ollama_wrapper.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.raise_for_status.return_value = None
            mock_get.return_value = mock_response
            
            with patch.object(LoggingMixin, 'log_provider_init') as mock_log:
                wrapper = OllamaWrapper(api_key=None, model="llama3", base_url="http://localhost:11434")
                
                assert wrapper.model == "llama3"
                assert wrapper.provider == "ollama"
                assert wrapper.base_url == "http://localhost:11434"
                assert wrapper.api_key is None
                mock_log.assert_called_once_with("ollama", "llama3")
                
                # Verify connection was verified
                mock_get.assert_called_once_with(
                    "http://localhost:11434/api/tags",
                    timeout=5
                )

    def test_ollama_wrapper_connection_error(self):
        """Test Ollama wrapper with connection error"""
        from llmwrapper.ollama_wrapper import OllamaWrapper
        import requests
        
        with patch('llmwrapper.ollama_wrapper.requests.get') as mock_get:
            mock_get.side_effect = requests.RequestException("Connection failed")
            
            with pytest.raises(ConnectionError, match="Ollama server not accessible"):
                OllamaWrapper(api_key=None, model="llama3")

class TestWrapperChatMethods:
    """Test the chat methods of each wrapper with proper mocking"""

    def test_openai_chat_method(self):
        """Test OpenAI chat method with logging"""
        from llmwrapper.openai_wrapper import OpenAIWrapper
        
        # Mock response structure
        mock_usage = Mock()
        mock_usage.model_dump.return_value = {
            "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30
        }
        
        mock_message = Mock()
        mock_message.content = "Test response"
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        with patch('llmwrapper.openai_wrapper.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            with patch.object(LoggingMixin, 'log_call_start') as mock_start, \
                 patch.object(LoggingMixin, 'log_call_end') as mock_end, \
                 patch.object(LoggingMixin, 'log_token_usage') as mock_usage_log, \
                 patch.object(LoggingMixin, 'log_provider_init'):
                
                wrapper = OpenAIWrapper(api_key="test-key", model="gpt-4")
                messages = [{"role": "user", "content": "Hello"}]
                response = wrapper.chat(messages)
                
                assert response == "Test response"
                mock_start.assert_called_once_with("openai", "gpt-4", 1)
                mock_end.assert_called_once()
                mock_usage_log.assert_called_once()

    def test_grok_chat_method(self):
        """Test Grok chat method with proper API mocking"""
        from llmwrapper.grok_wrapper import GrokWrapper
        
        # Mock response structure (same as OpenAI since Grok uses compatible API)
        mock_usage = Mock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        
        mock_message = Mock()
        mock_message.content = "Grok response"
        
        mock_choice = Mock()
        mock_choice.message = mock_message
        
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        with patch('llmwrapper.grok_wrapper.OpenAI') as mock_openai_class:
            mock_client = Mock()
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai_class.return_value = mock_client
            
            with patch.object(LoggingMixin, 'log_call_start') as mock_start, \
                 patch.object(LoggingMixin, 'log_call_end') as mock_end, \
                 patch.object(LoggingMixin, 'log_token_usage') as mock_usage_log, \
                 patch.object(LoggingMixin, 'log_provider_init'):
                
                wrapper = GrokWrapper(api_key="test-key", model="grok-beta")
                messages = [{"role": "user", "content": "Hello"}]
                response = wrapper.chat(messages)
                
                assert response == "Grok response"
                mock_start.assert_called_once_with("grok", "grok-beta", 1)
                mock_end.assert_called_once()
                mock_usage_log.assert_called_once()
                
                # Verify OpenAI client was initialized with correct parameters
                mock_openai_class.assert_called_once_with(
                    api_key="test-key",
                    base_url="https://api.x.ai/v1"
                )

    def test_ollama_chat_method(self):
        """Test Ollama chat method with proper API mocking"""
        from llmwrapper.ollama_wrapper import OllamaWrapper
        
        # Mock response structure from Ollama API
        mock_response_json = {
            "response": "Hello! I'm Llama, an AI assistant.",
            "prompt_eval_count": 10,
            "eval_count": 20
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_response_json
        mock_response.raise_for_status.return_value = None
        
        with patch('llmwrapper.ollama_wrapper.requests.get') as mock_get, \
             patch('llmwrapper.ollama_wrapper.requests.post') as mock_post:
            
            # Mock connection verification
            mock_get.return_value = Mock()
            mock_get.return_value.raise_for_status.return_value = None
            
            # Mock API call
            mock_post.return_value = mock_response
            
            with patch.object(LoggingMixin, 'log_call_start') as mock_start, \
                 patch.object(LoggingMixin, 'log_call_end') as mock_end, \
                 patch.object(LoggingMixin, 'log_token_usage') as mock_usage_log, \
                 patch.object(LoggingMixin, 'log_provider_init'):
                
                wrapper = OllamaWrapper(api_key=None, model="llama3", base_url="http://localhost:11434")
                messages = [{"role": "user", "content": "Hello"}]
                response = wrapper.chat(messages)
                
                assert response == "Hello! I'm Llama, an AI assistant."
                mock_start.assert_called_once_with("ollama", "llama3", 1)
                mock_end.assert_called_once()
                mock_usage_log.assert_called_once()
                
                # Verify the API call was made correctly
                mock_post.assert_called_once()
                call_args = mock_post.call_args
                assert call_args[0][0] == "http://localhost:11434/api/generate"
                assert call_args[1]["json"]["model"] == "llama3"
                assert call_args[1]["json"]["stream"] is False

    def test_ollama_list_models_method(self):
        """Test Ollama list_models method"""
        from llmwrapper.ollama_wrapper import OllamaWrapper
        
        # Mock response for list models API
        mock_models_response = {
            "models": [
                {"name": "llama3:latest"},
                {"name": "mistral:latest"},
                {"name": "codellama:latest"}
            ]
        }
        
        mock_response = Mock()
        mock_response.json.return_value = mock_models_response
        mock_response.raise_for_status.return_value = None
        
        with patch('llmwrapper.ollama_wrapper.requests.get') as mock_get:
            # First call for connection verification, second for list models
            mock_get.return_value = mock_response
            
            with patch.object(LoggingMixin, 'log_provider_init'):
                wrapper = OllamaWrapper(api_key=None, model="llama3", base_url="http://localhost:11434")
                models = wrapper.list_models()
                
                assert models == ["llama3:latest", "mistral:latest", "codellama:latest"]
                assert mock_get.call_count == 2  # Connection check + list models

    def test_ollama_message_conversion(self):
        """Test message conversion from OpenAI format to Ollama format"""
        from llmwrapper.ollama_wrapper import OllamaWrapper
        
        with patch('llmwrapper.ollama_wrapper.requests.get') as mock_get:
            mock_get.return_value = Mock()
            mock_get.return_value.raise_for_status.return_value = None
            
            with patch.object(LoggingMixin, 'log_provider_init'):
                wrapper = OllamaWrapper(api_key=None, model="llama3")
                
                messages = [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ]
                
                prompt = wrapper._convert_messages_to_prompt(messages)
                
                expected_prompt = ("System: You are helpful.\n\n"
                                 "Human: Hello\n\n"
                                 "Assistant: Hi there!\n\n"
                                 "Human: How are you?\n\n"
                                 "Assistant:")
                
                assert prompt == expected_prompt

class TestErrorHandling:
    """Test error handling scenarios"""
    
    def test_factory_with_missing_api_key(self):
        """Test factory behavior with missing API key"""
        config = {"model": "gpt-4"}  # Missing api_key
        
        with pytest.raises(KeyError):
            get_llm("openai", config)
    
    def test_factory_logging_on_instantiation(self):
        """Test that factory logs provider instantiation"""
        with patch('llmwrapper.registry.logger') as mock_logger, \
             patch('llmwrapper.factory.OpenAIWrapper') as mock_wrapper:
            
            config = {"api_key": "test", "model": "gpt-4"}
            get_llm("openai", config)
            
            mock_logger.info.assert_called_with("Instantiating OpenAIWrapper with model: gpt-4")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
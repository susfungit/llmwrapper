import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from llmwrapper import get_async_llm
from llmwrapper.async_openai_wrapper import AsyncOpenAIWrapper
from llmwrapper.async_anthropic_wrapper import AsyncClaudeWrapper
from llmwrapper.async_gemini_wrapper import AsyncGeminiWrapper
from llmwrapper.async_grok_wrapper import AsyncGrokWrapper
from llmwrapper.security_utils import SecurityUtils

class TestAsyncLLMWrappers:
    """Test the async LLM wrapper implementations"""

    def test_get_async_llm_openai(self):
        """Test async factory function for OpenAI"""
        config = {"api_key": "sk-test1234567890abcdef1234567890abcdef", "model": "gpt-4"}
        llm = get_async_llm("openai", config)
        
        assert isinstance(llm, AsyncOpenAIWrapper)
        assert llm.model == "gpt-4"
        assert llm.provider == "openai"

    def test_get_async_llm_anthropic(self):
        """Test async factory function for Anthropic"""
        config = {"api_key": "sk-ant-test1234567890abcdef1234567890abcdef", "model": "claude-3-sonnet-20240229"}
        llm = get_async_llm("anthropic", config)
        
        assert isinstance(llm, AsyncClaudeWrapper)
        assert llm.model == "claude-3-sonnet-20240229"
        assert llm.provider == "anthropic"

    def test_get_async_llm_gemini(self):
        """Test async factory function for Gemini"""
        config = {"api_key": "AIzatest1234567890abcdef1234567890abcdef", "model": "gemini-pro"}
        llm = get_async_llm("gemini", config)
        
        assert isinstance(llm, AsyncGeminiWrapper)
        assert llm.model == "gemini-pro"
        assert llm.provider == "gemini"

    def test_get_async_llm_grok(self):
        """Test async factory function for Grok"""
        config = {"api_key": "xai-test1234567890abcdef1234567890abcdef", "model": "grok-beta"}
        llm = get_async_llm("grok", config)
        
        assert isinstance(llm, AsyncGrokWrapper)
        assert llm.model == "grok-beta"
        assert llm.provider == "grok"

    def test_get_async_llm_ollama(self):
        """Test async factory function for Ollama"""
        from llmwrapper.async_ollama_wrapper import AsyncOllamaWrapper
        config = {"api_key": None, "model": "llama3", "base_url": "http://localhost:11434"}
        llm = get_async_llm("ollama", config)
        
        assert isinstance(llm, AsyncOllamaWrapper)
        assert llm.model == "llama3"
        assert llm.provider == "ollama"

    def test_get_async_llm_invalid_provider(self):
        """Test async factory with invalid provider"""
        config = {"api_key": "sk-test1234567890abcdef1234567890abcdef"}

        with pytest.raises(ValueError, match="Unsupported async provider"):
            get_async_llm("unknown", config)

    def test_async_openai_wrapper_init(self):
        """Test AsyncOpenAIWrapper initialization"""
        wrapper = AsyncOpenAIWrapper(api_key="sk-test1234567890abcdef1234567890abcdef", model="gpt-4")
        
        assert wrapper.model == "gpt-4"
        assert wrapper.provider == "openai"
        assert hasattr(wrapper, 'client')

    @pytest.mark.asyncio
    @patch('llmwrapper.async_openai_wrapper.AsyncOpenAI')
    async def test_async_openai_wrapper_chat(self, mock_openai):
        """Test AsyncOpenAIWrapper chat method"""
        # Mock the async response
        mock_usage = MagicMock()
        mock_usage.model_dump.return_value = {
            "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30
        }
        
        mock_message = MagicMock()
        mock_message.content = "Test async response"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        # Mock the client
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        wrapper = AsyncOpenAIWrapper(api_key="sk-test1234567890abcdef1234567890abcdef", model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]
        response = await wrapper.chat(messages)
        
        assert response == "Test async response"
        mock_client.chat.completions.create.assert_called_once()

    def test_async_claude_wrapper_init(self):
        """Test AsyncClaudeWrapper initialization"""
        wrapper = AsyncClaudeWrapper(api_key="sk-ant-test1234567890abcdef1234567890abcdef", model="claude-3-sonnet-20240229")
        
        assert wrapper.model == "claude-3-sonnet-20240229"
        assert wrapper.provider == "anthropic"
        assert hasattr(wrapper, 'client')

    @pytest.mark.asyncio
    @patch('llmwrapper.async_anthropic_wrapper.anthropic.AsyncAnthropic')
    async def test_async_claude_wrapper_chat(self, mock_anthropic):
        """Test AsyncClaudeWrapper chat method"""
        # Mock the async response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Test async claude response")]
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20
        
        mock_client = AsyncMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        wrapper = AsyncClaudeWrapper(api_key="sk-ant-test1234567890abcdef1234567890abcdef", model="claude-3-sonnet-20240229")
        messages = [{"role": "user", "content": "Hello"}]
        response = await wrapper.chat(messages)
        
        assert response == "Test async claude response"
        mock_client.messages.create.assert_called_once()

    def test_async_gemini_wrapper_init(self):
        """Test AsyncGeminiWrapper initialization"""
        wrapper = AsyncGeminiWrapper(api_key="AIzatest1234567890abcdef1234567890abcdef", model="gemini-pro")
        
        assert wrapper.model == "gemini-pro"
        assert wrapper.provider == "gemini"
        assert hasattr(wrapper, 'client')

    @pytest.mark.asyncio
    @patch('llmwrapper.async_gemini_wrapper.genai.Client')
    async def test_async_gemini_wrapper_chat(self, mock_genai):
        """Test AsyncGeminiWrapper chat method"""
        # Mock the response structure that matches Gemini API
        mock_part = MagicMock()
        mock_part.text = "Test async gemini response"
        
        mock_candidate = MagicMock()
        mock_candidate.content.parts = [mock_part]
        
        mock_response = MagicMock()
        mock_response.candidates = [mock_candidate]
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 10
        mock_response.usage_metadata.candidates_token_count = 20
        mock_response.usage_metadata.total_token_count = 30
        
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_genai.return_value = mock_client
        
        wrapper = AsyncGeminiWrapper(api_key="AIzatest1234567890abcdef1234567890abcdef", model="gemini-pro")
        messages = [{"role": "user", "content": "Hello"}]
        response = await wrapper.chat(messages)
        
        assert response == "Test async gemini response"

    def test_async_grok_wrapper_init(self):
        """Test AsyncGrokWrapper initialization"""
        wrapper = AsyncGrokWrapper(api_key="xai-test1234567890abcdef1234567890abcdef", model="grok-beta")
        
        assert wrapper.model == "grok-beta"
        assert wrapper.provider == "grok"
        assert hasattr(wrapper, 'client')

    def test_async_ollama_wrapper_init(self):
        """Test AsyncOllamaWrapper initialization"""
        from llmwrapper.async_ollama_wrapper import AsyncOllamaWrapper
        wrapper = AsyncOllamaWrapper(api_key=None, model="llama3", base_url="http://localhost:11434")
        
        assert wrapper.model == "llama3"
        assert wrapper.provider == "ollama"
        assert wrapper.base_url == "http://localhost:11434"
        assert wrapper.api_key is None
        assert wrapper._session is None

    @pytest.mark.asyncio
    @patch('llmwrapper.async_grok_wrapper.AsyncOpenAI')
    async def test_async_grok_wrapper_chat(self, mock_openai):
        """Test AsyncGrokWrapper chat method"""
        # Mock the async response
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 30
        
        mock_message = MagicMock()
        mock_message.content = "Test async grok response"
        
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        wrapper = AsyncGrokWrapper(api_key="xai-test1234567890abcdef1234567890abcdef", model="grok-beta")
        messages = [{"role": "user", "content": "Hello"}]
        response = await wrapper.chat(messages)
        
        assert response == "Test async grok response"
        mock_client.chat.completions.create.assert_called_once()

    @pytest.mark.skip(reason="Async context manager mocking is complex - tested via integration tests")
    @pytest.mark.asyncio
    async def test_async_ollama_wrapper_chat(self):
        """Test AsyncOllamaWrapper chat method"""
        # This test is skipped because mocking aiohttp with proper async context managers
        # is complex and we test this functionality via integration tests instead
        pass

    @pytest.mark.skip(reason="Async context manager mocking is complex - tested via integration tests")
    @pytest.mark.asyncio
    async def test_async_ollama_wrapper_list_models(self):
        """Test AsyncOllamaWrapper list_models method"""
        # This test is skipped because mocking aiohttp with proper async context managers
        # is complex and we test this functionality via integration tests instead
        pass

    @pytest.mark.asyncio
    async def test_async_ollama_wrapper_context_manager(self):
        """Test AsyncOllamaWrapper as async context manager"""
        from llmwrapper.async_ollama_wrapper import AsyncOllamaWrapper
        wrapper = AsyncOllamaWrapper(api_key=None, model="llama3", base_url="http://localhost:11434")
        
        async with wrapper as ctx_wrapper:
            assert ctx_wrapper is wrapper
        
        # After exiting context, session should be None or closed
        assert wrapper._session is None or wrapper._session.closed

    @pytest.mark.asyncio
    @patch('llmwrapper.async_anthropic_wrapper.anthropic.AsyncAnthropic')
    @patch('llmwrapper.async_openai_wrapper.AsyncOpenAI')
    async def test_concurrent_async_calls(self, mock_openai, mock_anthropic):
        """Test concurrent async calls to multiple providers"""
        # Mock OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock(message=MagicMock(content="OpenAI response"))]
        mock_openai_response.usage = MagicMock()
        mock_openai_response.usage.model_dump.return_value = {"total_tokens": 30}
        
        mock_openai_client = AsyncMock()
        mock_openai_client.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_openai_client
        
        # Mock Anthropic response
        mock_anthropic_response = MagicMock()
        mock_anthropic_response.content = [MagicMock(text="Anthropic response")]
        mock_anthropic_response.usage = MagicMock()
        mock_anthropic_response.usage.input_tokens = 10
        mock_anthropic_response.usage.output_tokens = 20
        
        mock_anthropic_client = AsyncMock()
        mock_anthropic_client.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_anthropic_client
        
        # Create wrappers
        openai_wrapper = AsyncOpenAIWrapper(api_key="sk-test1234567890abcdef1234567890abcdef", model="gpt-4")
        anthropic_wrapper = AsyncClaudeWrapper(api_key="sk-ant-test1234567890abcdef1234567890abcdef", model="claude-3-sonnet-20240229")
        
        messages = [{"role": "user", "content": "Hello"}]
        
        # Run concurrent calls
        results = await asyncio.gather(
            openai_wrapper.chat(messages),
            anthropic_wrapper.chat(messages)
        )
        
        assert results[0] == "OpenAI response"
        assert results[1] == "Anthropic response"

    @pytest.mark.asyncio
    @patch('llmwrapper.async_openai_wrapper.AsyncOpenAI')
    async def test_async_error_handling(self, mock_openai):
        """Test async error handling"""
        mock_client = AsyncMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client
        
        wrapper = AsyncOpenAIWrapper(api_key="sk-test1234567890abcdef1234567890abcdef", model="gpt-4")
        messages = [{"role": "user", "content": "Hello"}]
        
        with pytest.raises(Exception, match="API Error"):
            await wrapper.chat(messages)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
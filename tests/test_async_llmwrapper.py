import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from async_factory import get_async_llm
from async_openai_wrapper import AsyncOpenAIWrapper
from async_anthropic_wrapper import AsyncClaudeWrapper
from async_gemini_wrapper import AsyncGeminiWrapper
from async_grok_wrapper import AsyncGrokWrapper

class TestAsyncLLMWrappers:
    """Test suite for async LLM wrappers"""

    @pytest.fixture
    def openai_config(self):
        return {"api_key": "test_openai_key", "model": "gpt-4"}

    @pytest.fixture
    def anthropic_config(self):
        return {"api_key": "test_anthropic_key", "model": "claude-3-opus-20240229"}

    @pytest.fixture
    def gemini_config(self):
        return {"api_key": "test_gemini_key", "model": "gemini-pro"}

    @pytest.fixture
    def grok_config(self):
        return {"api_key": "test_grok_key", "model": "grok-beta", "base_url": "https://api.x.ai/v1"}

    @pytest.fixture
    def sample_messages(self):
        return [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]

    # Test Factory Function
    def test_get_async_llm_openai(self, openai_config):
        """Test factory creates AsyncOpenAIWrapper correctly"""
        llm = get_async_llm("openai", openai_config)
        assert isinstance(llm, AsyncOpenAIWrapper)
        assert llm.model == "gpt-4"

    def test_get_async_llm_anthropic(self, anthropic_config):
        """Test factory creates AsyncClaudeWrapper correctly"""
        llm = get_async_llm("anthropic", anthropic_config)
        assert isinstance(llm, AsyncClaudeWrapper)
        assert llm.model == "claude-3-opus-20240229"

    def test_get_async_llm_gemini(self, gemini_config):
        """Test factory creates AsyncGeminiWrapper correctly"""
        llm = get_async_llm("gemini", gemini_config)
        assert isinstance(llm, AsyncGeminiWrapper)
        assert llm.model == "gemini-pro"

    def test_get_async_llm_grok(self, grok_config):
        """Test factory creates AsyncGrokWrapper correctly"""
        llm = get_async_llm("grok", grok_config)
        assert isinstance(llm, AsyncGrokWrapper)
        assert llm.model == "grok-beta"

    def test_get_async_llm_invalid_provider(self, openai_config):
        """Test factory raises error for invalid provider"""
        with pytest.raises(ValueError, match="Unsupported async LLM provider"):
            get_async_llm("invalid_provider", openai_config)

    # Test AsyncOpenAIWrapper
    @pytest.mark.asyncio
    async def test_async_openai_wrapper_init(self, openai_config):
        """Test AsyncOpenAIWrapper initialization"""
        wrapper = AsyncOpenAIWrapper(api_key=openai_config["api_key"], model=openai_config["model"])
        assert wrapper.model == "gpt-4"
        assert wrapper.provider == "openai"

    @pytest.mark.asyncio
    @patch('async_openai_wrapper.AsyncOpenAI')
    async def test_async_openai_wrapper_chat(self, mock_openai_client, openai_config, sample_messages):
        """Test AsyncOpenAIWrapper chat method"""
        # Mock the client and response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! I'm doing well, thank you."
        mock_response.usage = MagicMock()
        mock_response.usage.model_dump.return_value = {
            'prompt_tokens': 10, 'completion_tokens': 8, 'total_tokens': 18
        }
        
        mock_client_instance = AsyncMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance
        
        wrapper = AsyncOpenAIWrapper(api_key=openai_config["api_key"], model=openai_config["model"])
        response = await wrapper.chat(sample_messages)
        
        assert response == "Hello! I'm doing well, thank you."
        mock_client_instance.chat.completions.create.assert_called_once()

    # Test AsyncClaudeWrapper
    @pytest.mark.asyncio
    async def test_async_claude_wrapper_init(self, anthropic_config):
        """Test AsyncClaudeWrapper initialization"""
        wrapper = AsyncClaudeWrapper(api_key=anthropic_config["api_key"], model=anthropic_config["model"])
        assert wrapper.model == "claude-3-opus-20240229"
        assert wrapper.provider == "anthropic"

    @pytest.mark.asyncio
    @patch('async_anthropic_wrapper.anthropic.AsyncAnthropic')
    async def test_async_claude_wrapper_chat(self, mock_anthropic_client, anthropic_config, sample_messages):
        """Test AsyncClaudeWrapper chat method"""
        # Mock the client and response
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Hello! I'm Claude, nice to meet you."
        mock_response.usage = MagicMock()
        mock_response.usage.input_tokens = 12
        mock_response.usage.output_tokens = 9
        
        mock_client_instance = AsyncMock()
        mock_client_instance.messages.create.return_value = mock_response
        mock_anthropic_client.return_value = mock_client_instance
        
        wrapper = AsyncClaudeWrapper(api_key=anthropic_config["api_key"], model=anthropic_config["model"])
        response = await wrapper.chat(sample_messages)
        
        assert response == "Hello! I'm Claude, nice to meet you."
        mock_client_instance.messages.create.assert_called_once()

    # Test AsyncGeminiWrapper
    @pytest.mark.asyncio
    async def test_async_gemini_wrapper_init(self, gemini_config):
        """Test AsyncGeminiWrapper initialization"""
        wrapper = AsyncGeminiWrapper(api_key=gemini_config["api_key"], model=gemini_config["model"])
        assert wrapper.model == "gemini-pro"
        assert wrapper.provider == "gemini"

    @pytest.mark.asyncio
    @patch('async_gemini_wrapper.genai.Client')
    @patch('asyncio.get_event_loop')
    async def test_async_gemini_wrapper_chat(self, mock_get_loop, mock_genai_client, gemini_config, sample_messages):
        """Test AsyncGeminiWrapper chat method"""
        # Mock the response
        mock_response = MagicMock()
        mock_response.candidates = [MagicMock()]
        mock_response.candidates[0].content.parts = [MagicMock()]
        mock_response.candidates[0].content.parts[0].text = "Hello! I'm Gemini."
        mock_response.usage_metadata = MagicMock()
        mock_response.usage_metadata.prompt_token_count = 8
        mock_response.usage_metadata.candidates_token_count = 5
        mock_response.usage_metadata.total_token_count = 13
        
        # Mock the client
        mock_client_instance = MagicMock()
        mock_client_instance.models.generate_content.return_value = mock_response
        mock_genai_client.return_value = mock_client_instance
        
        # Mock the event loop and executor
        mock_loop = AsyncMock()
        mock_loop.run_in_executor.return_value = mock_response
        mock_get_loop.return_value = mock_loop
        
        wrapper = AsyncGeminiWrapper(api_key=gemini_config["api_key"], model=gemini_config["model"])
        response = await wrapper.chat(sample_messages)
        
        assert response == "Hello! I'm Gemini."
        mock_loop.run_in_executor.assert_called_once()

    # Test AsyncGrokWrapper
    @pytest.mark.asyncio
    async def test_async_grok_wrapper_init(self, grok_config):
        """Test AsyncGrokWrapper initialization"""
        wrapper = AsyncGrokWrapper(
            api_key=grok_config["api_key"], 
            model=grok_config["model"],
            base_url=grok_config["base_url"]
        )
        assert wrapper.model == "grok-beta"
        assert wrapper.provider == "grok"
        assert wrapper.base_url == "https://api.x.ai/v1"

    @pytest.mark.asyncio
    @patch('async_grok_wrapper.AsyncOpenAI')
    async def test_async_grok_wrapper_chat(self, mock_openai_client, grok_config, sample_messages):
        """Test AsyncGrokWrapper chat method"""
        # Mock the client and response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! I'm Grok."
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 11
        mock_response.usage.completion_tokens = 4
        mock_response.usage.total_tokens = 15
        
        mock_client_instance = AsyncMock()
        mock_client_instance.chat.completions.create.return_value = mock_response
        mock_openai_client.return_value = mock_client_instance
        
        wrapper = AsyncGrokWrapper(
            api_key=grok_config["api_key"], 
            model=grok_config["model"],
            base_url=grok_config["base_url"]
        )
        response = await wrapper.chat(sample_messages)
        
        assert response == "Hello! I'm Grok."
        mock_client_instance.chat.completions.create.assert_called_once()

    # Test Concurrent Operations
    @pytest.mark.asyncio
    @patch('async_openai_wrapper.AsyncOpenAI')
    @patch('async_anthropic_wrapper.anthropic.AsyncAnthropic')
    async def test_concurrent_async_calls(self, mock_anthropic, mock_openai, openai_config, anthropic_config, sample_messages):
        """Test concurrent async calls to multiple providers"""
        # Mock OpenAI response
        mock_openai_response = MagicMock()
        mock_openai_response.choices = [MagicMock()]
        mock_openai_response.choices[0].message.content = "OpenAI response"
        mock_openai_response.usage = MagicMock()
        mock_openai_response.usage.model_dump.return_value = {}
        
        mock_openai_instance = AsyncMock()
        mock_openai_instance.chat.completions.create.return_value = mock_openai_response
        mock_openai.return_value = mock_openai_instance
        
        # Mock Anthropic response
        mock_anthropic_response = MagicMock()
        mock_anthropic_response.content = [MagicMock()]
        mock_anthropic_response.content[0].text = "Anthropic response"
        mock_anthropic_response.usage = None
        
        mock_anthropic_instance = AsyncMock()
        mock_anthropic_instance.messages.create.return_value = mock_anthropic_response
        mock_anthropic.return_value = mock_anthropic_instance
        
        # Create wrappers
        openai_wrapper = AsyncOpenAIWrapper(api_key=openai_config["api_key"], model=openai_config["model"])
        anthropic_wrapper = AsyncClaudeWrapper(api_key=anthropic_config["api_key"], model=anthropic_config["model"])
        
        # Make concurrent calls
        openai_task = openai_wrapper.chat(sample_messages)
        anthropic_task = anthropic_wrapper.chat(sample_messages)
        
        openai_response, anthropic_response = await asyncio.gather(openai_task, anthropic_task)
        
        assert openai_response == "OpenAI response"
        assert anthropic_response == "Anthropic response"

    # Test Error Handling
    @pytest.mark.asyncio
    @patch('async_openai_wrapper.AsyncOpenAI')
    async def test_async_error_handling(self, mock_openai_client, openai_config, sample_messages):
        """Test error handling in async operations"""
        mock_client_instance = AsyncMock()
        mock_client_instance.chat.completions.create.side_effect = Exception("API Error")
        mock_openai_client.return_value = mock_client_instance
        
        wrapper = AsyncOpenAIWrapper(api_key=openai_config["api_key"], model=openai_config["model"])
        
        with pytest.raises(Exception, match="API Error"):
            await wrapper.chat(sample_messages)

if __name__ == "__main__":
    pytest.main([__file__])
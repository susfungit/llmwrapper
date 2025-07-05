import aiohttp
import asyncio
from typing import Dict, Any, Optional
from .async_base import AsyncBaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin
from .registry import register_async_provider

@register_async_provider("ollama", "llama3", base_url="http://localhost:11434")
class AsyncOllamaWrapper(AsyncBaseLLM, LoggingMixin):
    """
    Async Ollama wrapper for local LLM inference.
    
    Ollama provides a simple HTTP API for running local LLMs like Llama, Mistral, etc.
    Models are managed externally through the Ollama CLI.
    """
    
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434", 
                 api_key: str = None):
        """
        Initialize async Ollama wrapper.
        
        Args:
            model: Model name (e.g., "llama3", "mistral", "codellama")
            base_url: Ollama server URL
            api_key: Unused but kept for interface consistency
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.provider = "ollama"
        self.api_key = api_key  # Stored but not used - for interface consistency
        self._session = None
        
        self.log_provider_init(self.provider, self.model)
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _verify_connection(self) -> None:
        """Verify that Ollama server is accessible."""
        session = await self._get_session()
        try:
            async with session.get(f"{self.base_url}/api/tags", timeout=aiohttp.ClientTimeout(total=5)) as response:
                response.raise_for_status()
                logger.debug(f"Ollama server accessible at {self.base_url}")
        except aiohttp.ClientError as e:
            logger.error(f"Failed to connect to Ollama server at {self.base_url}: {e}")
            raise ConnectionError(f"Ollama server not accessible at {self.base_url}. "
                                f"Please ensure Ollama is running: 'ollama serve'")
    
    def _convert_messages_to_prompt(self, messages: list[dict]) -> str:
        """
        Convert OpenAI-style messages to a single prompt string.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"Human: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
            else:
                prompt_parts.append(f"{role}: {content}")
        
        # Add final prompt for assistant response
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Send chat messages to Ollama and return response.
        
        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Response content as string
        """
        start = self.log_call_start(self.provider, self.model, len(messages))
        
        # Verify connection on first use
        await self._verify_connection()
        
        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)
        
        # Prepare request payload
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {}
        }
        
        # Map common parameters to Ollama options
        if 'temperature' in kwargs:
            payload['options']['temperature'] = kwargs['temperature']
        if 'max_tokens' in kwargs:
            payload['options']['num_predict'] = kwargs['max_tokens']
        if 'top_p' in kwargs:
            payload['options']['top_p'] = kwargs['top_p']
        if 'top_k' in kwargs:
            payload['options']['top_k'] = kwargs['top_k']
        
        session = await self._get_session()
        
        try:
            timeout = aiohttp.ClientTimeout(total=kwargs.get('timeout', 120))
            async with session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=timeout
            ) as response:
                response.raise_for_status()
                result = await response.json()
                
                if 'response' not in result:
                    raise ValueError(f"Unexpected response format from Ollama: {result}")
                
                content = result['response'].strip()
                
                # Log token usage if available
                if 'eval_count' in result:
                    token_info = {
                        'prompt_tokens': result.get('prompt_eval_count', 0),
                        'completion_tokens': result.get('eval_count', 0),
                        'total_tokens': result.get('prompt_eval_count', 0) + result.get('eval_count', 0)
                    }
                    self.log_token_usage(self.provider, token_info)
                
                self.log_call_end(self.provider, self.model, start)
                return content
                
        except aiohttp.ClientError as e:
            logger.error(f"Ollama API request failed: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in Ollama chat: {e}")
            raise
    
    async def list_models(self) -> list:
        """
        List available models on the Ollama server.
        
        Returns:
            List of model names
        """
        session = await self._get_session()
        
        try:
            async with session.get(f"{self.base_url}/api/tags") as response:
                response.raise_for_status()
                result = await response.json()
                models = [model['name'] for model in result.get('models', [])]
                
                logger.debug(f"Available Ollama models: {models}")
                return models
                
        except aiohttp.ClientError as e:
            logger.error(f"Failed to list Ollama models: {e}")
            raise RuntimeError(f"Failed to list Ollama models: {e}")
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.debug("Ollama session closed")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close() 
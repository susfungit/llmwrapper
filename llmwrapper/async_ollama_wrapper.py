import aiohttp
import asyncio
from typing import Dict, Any, Optional
from .async_base import AsyncBaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin
from .registry import register_async_provider
from .security_utils import SecurityUtils

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
        # Validate base URL format
        if not SecurityUtils.validate_url(base_url):
            self.log_security_event("INVALID_URL", {
                "provider": "ollama",
                "base_url": "invalid_format"
            })
            raise ValueError("Invalid Ollama base URL format")
        
        # Validate API key (should be None for Ollama)
        if not SecurityUtils.validate_api_key(api_key, "ollama"):
            self.log_security_event("INVALID_API_KEY", {
                "provider": "ollama",
                "api_key_format": "invalid"
            })
            raise ValueError("Invalid API key for Ollama (should be None)")
        
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
            self.log_security_event("CONNECTION_FAILED", {
                "provider": self.provider,
                "base_url": self.base_url,
                "error_type": type(e).__name__
            })
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
        # Validate messages before processing
        if not SecurityUtils.validate_messages(messages):
            self.log_security_event("INVALID_MESSAGES", {
                "provider": self.provider,
                "message_count": len(messages) if isinstance(messages, list) else 0
            })
            raise ValueError("Invalid message format or potential injection detected")
        
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
        
        # Map common parameters to Ollama options with validation
        if 'temperature' in kwargs:
            temp = kwargs['temperature']
            if not isinstance(temp, (int, float)) or temp < 0 or temp > 2:
                self.log_security_event("INVALID_PARAMETER", {
                    "provider": self.provider,
                    "parameter": "temperature",
                    "value": temp
                })
                raise ValueError("Temperature must be between 0 and 2")
            payload['options']['temperature'] = temp
            
        if 'max_tokens' in kwargs:
            max_tokens = kwargs['max_tokens']
            if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > 32768:
                self.log_security_event("INVALID_PARAMETER", {
                    "provider": self.provider,
                    "parameter": "max_tokens",
                    "value": max_tokens
                })
                raise ValueError("max_tokens must be between 1 and 32768")
            payload['options']['num_predict'] = max_tokens
            
        if 'top_p' in kwargs:
            top_p = kwargs['top_p']
            if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > 1:
                self.log_security_event("INVALID_PARAMETER", {
                    "provider": self.provider,
                    "parameter": "top_p",
                    "value": top_p
                })
                raise ValueError("top_p must be between 0 and 1")
            payload['options']['top_p'] = top_p
            
        if 'top_k' in kwargs:
            top_k = kwargs['top_k']
            if not isinstance(top_k, int) or top_k < 1 or top_k > 100:
                self.log_security_event("INVALID_PARAMETER", {
                    "provider": self.provider,
                    "parameter": "top_k",
                    "value": top_k
                })
                raise ValueError("top_k must be between 1 and 100")
            payload['options']['top_k'] = top_k
        
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
                    self.log_security_event("UNEXPECTED_RESPONSE", {
                        "provider": self.provider,
                        "response_keys": list(result.keys()) if isinstance(result, dict) else "non_dict"
                    })
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
            self.log_security_event("API_REQUEST_FAILED", {
                "provider": self.provider,
                "error_type": type(e).__name__,
                "base_url": self.base_url
            })
            logger.error(f"Ollama API request failed: {e}")
            raise RuntimeError(f"Failed to get response from Ollama: {e}")
        except Exception as e:
            self.log_security_event("UNEXPECTED_ERROR", {
                "provider": self.provider,
                "error_type": type(e).__name__,
                "model": self.model
            })
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
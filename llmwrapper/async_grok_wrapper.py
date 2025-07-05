from .async_base import AsyncBaseLLM
from .logging_mixin import LoggingMixin
from .registry import register_async_provider
from .security_utils import SecurityUtils
from openai import AsyncOpenAI

@register_async_provider("grok", "grok-beta", base_url="https://api.x.ai/v1")
class AsyncGrokWrapper(AsyncBaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "grok-beta", base_url: str = "https://api.x.ai/v1"):
        """
        Initialize async Grok wrapper with xAI API credentials.
        
        Args:
            api_key: xAI API key
            model: Grok model name (default: grok-beta)
            base_url: xAI API base URL (default: https://api.x.ai/v1)
        """
        # Validate API key format
        if not SecurityUtils.validate_api_key(api_key, "grok"):
            self.log_security_event("INVALID_API_KEY", {
                "provider": "grok",
                "api_key_format": "invalid"
            })
            raise ValueError("Invalid Grok API key format")
        
        # Validate base URL format
        if not SecurityUtils.validate_url(base_url):
            self.log_security_event("INVALID_URL", {
                "provider": "grok",
                "base_url": "invalid_format"
            })
            raise ValueError("Invalid Grok base URL format")
        
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.provider = "grok"
        
        # Initialize AsyncOpenAI client with xAI endpoint (Grok uses OpenAI-compatible API)
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )
        
        self.log_provider_init(self.provider, self.model)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        """
        Send async chat request to Grok API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters to pass to the API
            
        Returns:
            str: Generated response content
            
        Raises:
            Exception: If API call fails
        """
        # Validate messages before processing
        if not SecurityUtils.validate_messages(messages):
            self.log_security_event("INVALID_MESSAGES", {
                "provider": self.provider,
                "message_count": len(messages) if isinstance(messages, list) else 0
            })
            raise ValueError("Invalid message format or potential injection detected")
        
        start = self.log_call_start(self.provider, self.model, len(messages))
        
        try:
            # Call Grok API using AsyncOpenAI-compatible interface
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            
            self.log_call_end(self.provider, self.model, start)
            
            # Log token usage if available
            usage_info = {}
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            self.log_token_usage(self.provider, usage_info)
            
            return response.choices[0].message.content
            
        except Exception as e:
            self.log_security_event("API_ERROR", {
                "provider": self.provider,
                "error_type": type(e).__name__,
                "model": self.model
            })
            raise 
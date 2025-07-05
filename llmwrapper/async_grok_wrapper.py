from .async_base import AsyncBaseLLM
from .logging_mixin import LoggingMixin
from openai import AsyncOpenAI

class AsyncGrokWrapper(AsyncBaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "grok-beta", base_url: str = "https://api.x.ai/v1"):
        """
        Initialize async Grok wrapper with xAI API credentials.
        
        Args:
            api_key: xAI API key
            model: Grok model name (default: grok-beta)
            base_url: xAI API base URL (default: https://api.x.ai/v1)
        """
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
            self.log_call_end(self.provider, self.model, start)
            self.log_token_usage(self.provider, {})
            raise Exception(f"Grok API error: {str(e)}") 
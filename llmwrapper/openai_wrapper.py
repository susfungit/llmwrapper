from openai import OpenAI
from .base import BaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin
from .registry import register_sync_provider
from .security_utils import SecurityUtils

@register_sync_provider("openai", "gpt-4")
class OpenAIWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        # Validate API key format
        if not SecurityUtils.validate_api_key(api_key, "openai"):
            self.log_security_event("INVALID_API_KEY", {
                "provider": "openai",
                "api_key_format": "invalid"
            })
            raise ValueError("Invalid OpenAI API key format")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.provider = "openai"
        self.log_provider_init(self.provider, self.model)

    def chat(self, messages: list[dict], **kwargs) -> str:
        # Validate messages before sending
        if not SecurityUtils.validate_messages(messages):
            self.log_security_event("INVALID_MESSAGES", {
                "provider": self.provider,
                "message_count": len(messages) if isinstance(messages, list) else 0
            })
            raise ValueError("Invalid message format or potential injection detected")
        
        start = self.log_call_start(self.provider, self.model, len(messages))
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs
            )
            self.log_call_end(self.provider, self.model, start)
            self.log_token_usage(self.provider, response.usage.model_dump() if response.usage else {})
            return response.choices[0].message.content
        except Exception as e:
            self.log_security_event("API_ERROR", {
                "provider": self.provider,
                "error_type": type(e).__name__,
                "model": self.model
            })
            raise
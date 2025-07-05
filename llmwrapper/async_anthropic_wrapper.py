import anthropic
from .async_base import AsyncBaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin
from .registry import register_async_provider
from .security_utils import SecurityUtils

@register_async_provider("anthropic", "claude-3-opus-20240229")
class AsyncClaudeWrapper(AsyncBaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        # Validate API key format
        if not SecurityUtils.validate_api_key(api_key, "anthropic"):
            self.log_security_event("INVALID_API_KEY", {
                "provider": "anthropic",
                "api_key_format": "invalid"
            })
            raise ValueError("Invalid Anthropic API key format")
        
        self.client = anthropic.AsyncAnthropic(api_key=api_key)
        self.model = model
        self.provider = "anthropic"
        self.log_provider_init(self.provider, self.model)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        # Validate messages before processing
        if not SecurityUtils.validate_messages(messages):
            self.log_security_event("INVALID_MESSAGES", {
                "provider": self.provider,
                "message_count": len(messages) if isinstance(messages, list) else 0
            })
            raise ValueError("Invalid message format or potential injection detected")
        
        start = self.log_call_start(self.provider, self.model, len(messages))
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt += msg["content"] + "\n"
        
        try:
            response = await self.client.messages.create(
                model=self.model,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                **kwargs
            )
            self.log_call_end(self.provider, self.model, start)
            
            # Log token usage if available (Anthropic provides usage info)
            usage_info = {}
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    'prompt_tokens': response.usage.input_tokens,
                    'completion_tokens': response.usage.output_tokens,
                    'total_tokens': response.usage.input_tokens + response.usage.output_tokens
                }
            self.log_token_usage(self.provider, usage_info)
            
            return response.content[0].text
        except Exception as e:
            self.log_security_event("API_ERROR", {
                "provider": self.provider,
                "error_type": type(e).__name__,
                "model": self.model
            })
            raise 
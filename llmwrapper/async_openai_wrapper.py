from openai import AsyncOpenAI
from .async_base import AsyncBaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin
from .registry import register_async_provider

@register_async_provider("openai", "gpt-4")
class AsyncOpenAIWrapper(AsyncBaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.provider = "openai"
        self.log_provider_init(self.provider, self.model)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        start = self.log_call_start(self.provider, self.model, len(messages))
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        self.log_call_end(self.provider, self.model, start)
        self.log_token_usage(self.provider, response.usage.model_dump() if response.usage else {})
        return response.choices[0].message.content 
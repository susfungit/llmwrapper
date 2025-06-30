import openai
from .base import BaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin

class OpenAIWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        start = self.log_call_start(self.model, len(messages))
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        self.log_call_end(self.model, start)
        self.log_token_usage(response.get("usage", {}))
        return response["choices"][0]["message"]["content"]
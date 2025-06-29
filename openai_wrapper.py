import openai
from .base import BaseLLM

class OpenAIWrapper(BaseLLM):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        openai.api_key = api_key
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        return response["choices"][0]["message"]["content"]
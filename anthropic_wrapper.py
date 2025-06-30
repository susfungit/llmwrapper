import anthropic
from .base import BaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin

class ClaudeWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        self.log_call_start(self.model, len(messages))
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt += msg["content"] + "\n"
        start = self.log_call_start(self.model, len(messages))
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **kwargs
        )
        self.log_call_end(self.model, start)
        return response.content[0].text
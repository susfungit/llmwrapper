import anthropic
from .base import BaseLLM

class ClaudeWrapper(BaseLLM):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def chat(self, messages: list[dict], **kwargs) -> str:
        system_prompt = ""
        user_prompt = ""
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_prompt += msg["content"] + "\n"

        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            **kwargs
        )
        return response.content[0].text
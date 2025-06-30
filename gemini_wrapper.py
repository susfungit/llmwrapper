import google.generativeai as genai
from base import BaseLLM
from logging_mixin import LoggingMixin

class GeminiWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = model
        self.chat_model = genai.GenerativeModel(model)

    def chat(self, messages: list[dict], **kwargs) -> str:
        prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
        start = self.log_call_start(self.model, len(messages))
        response = self.chat_model.generate_content(prompt, **kwargs)
        self.log_call_end(self.model, start)
        return response.text
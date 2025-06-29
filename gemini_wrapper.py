import google.generativeai as genai
from .base import BaseLLM

class GeminiWrapper(BaseLLM):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        genai.configure(api_key=api_key)
        self.model = model
        self.chat_model = genai.GenerativeModel(model)

    def chat(self, messages: list[dict], **kwargs) -> str:
        history = []
        for msg in messages:
            if msg["role"] == "user":
                history.append(msg["content"])
        response = self.chat_model.generate_content("\n".join(history), **kwargs)
        return response.text
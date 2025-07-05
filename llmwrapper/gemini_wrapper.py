from google import genai
from google.genai import types

from .base import BaseLLM
from .logging_mixin import LoggingMixin
from .registry import register_sync_provider

@register_sync_provider("gemini", "gemini-pro")
class GeminiWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.provider = "gemini"
        self.log_provider_init(self.provider, self.model)

    def chat(self, messages: list[dict], **kwargs) -> str:
        prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
        start = self.log_call_start(self.provider, self.model, len(messages))
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_budget=0)),
            **kwargs
        )
        self.log_call_end(self.provider, self.model, start)
        
        # Log token usage if available (Gemini may provide usage info in some cases)
        usage_info = {}
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage_info = {
                'prompt_tokens': response.usage_metadata.prompt_token_count,
                'completion_tokens': response.usage_metadata.candidates_token_count,
                'total_tokens': response.usage_metadata.total_token_count
            }
        self.log_token_usage(self.provider, usage_info)
        
        return response.candidates[0].content.parts[0].text
import asyncio
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

from .async_base import AsyncBaseLLM
from .logging_mixin import LoggingMixin
from .registry import register_async_provider
from .security_utils import SecurityUtils

@register_async_provider("gemini", "gemini-pro")
class AsyncGeminiWrapper(AsyncBaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        # Validate API key format
        if not SecurityUtils.validate_api_key(api_key, "gemini"):
            self.log_security_event("INVALID_API_KEY", {
                "provider": "gemini",
                "api_key_format": "invalid"
            })
            raise ValueError("Invalid Gemini API key format")
        
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.provider = "gemini"
        self.log_provider_init(self.provider, self.model)

    async def chat(self, messages: list[dict], **kwargs) -> str:
        # Validate messages before processing
        if not SecurityUtils.validate_messages(messages):
            self.log_security_event("INVALID_MESSAGES", {
                "provider": self.provider,
                "message_count": len(messages) if isinstance(messages, list) else 0
            })
            raise ValueError("Invalid message format or potential injection detected")
        
        prompt = "\n".join([msg["content"] for msg in messages if msg["role"] == "user"])
        start = self.log_call_start(self.provider, self.model, len(messages))
        
        try:
            # Since google-genai doesn't have native async support yet, 
            # we run the sync operation in a thread pool to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                response = await loop.run_in_executor(
                    executor,
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=0)),
                        **kwargs
                    )
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
        except Exception as e:
            self.log_security_event("API_ERROR", {
                "provider": self.provider,
                "error_type": type(e).__name__,
                "model": self.model
            })
            raise 
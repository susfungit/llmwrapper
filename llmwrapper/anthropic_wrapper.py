import anthropic
from .base import BaseLLM
from .logger import logger
from .logging_mixin import LoggingMixin

class ClaudeWrapper(BaseLLM, LoggingMixin):
    def __init__(self, api_key: str, model: str = "claude-3-opus-20240229"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.provider = "anthropic"
        self.log_provider_init(self.provider, self.model)

    def chat(self, messages: list[dict], **kwargs) -> str:
        start = self.log_call_start(self.provider, self.model, len(messages))
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
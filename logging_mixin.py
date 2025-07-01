import time
from logger import logger

class LoggingMixin:
    def log_call_start(self, provider: str, model_name: str, message_count: int):
        logger.info(f"Calling {provider}/{model_name} with {message_count} message(s)")
        return time.time()

    def log_call_end(self, provider: str, model_name: str, start_time: float):
        elapsed = time.time() - start_time
        logger.info(f"{provider}/{model_name} response received in {elapsed:.2f} seconds")

    def log_token_usage(self, provider: str, usage: dict):
        if usage:
            logger.info(f"{provider} - Prompt tokens: {usage.get('prompt_tokens')}, "
                        f"Completion tokens: {usage.get('completion_tokens')}, "
                        f"Total: {usage.get('total_tokens')}")
        else:
            logger.warning(f"{provider} - Token usage information not available.")
            
    def log_provider_init(self, provider: str, model: str):
        logger.info(f"Initialized {provider} wrapper with model: {model}")
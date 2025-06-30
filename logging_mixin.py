import time
from .logger import logger

class LoggingMixin:
    def log_call_start(self, model_name: str, message_count: int):
        logger.info(f"Calling {model_name} with {message_count} message(s)")
        return time.time()

    def log_call_end(self, model_name: str, start_time: float):
        elapsed = time.time() - start_time
        logger.info(f"{model_name} response received in {elapsed:.2f} seconds")

    def log_token_usage(self, usage: dict):
        if usage:
            logger.info(f"Prompt tokens: {usage.get('prompt_tokens')}, "
                        f"Completion tokens: {usage.get('completion_tokens')}, "
                        f"Total: {usage.get('total_tokens')}")
        else:
            logger.warning("Token usage information not available.")
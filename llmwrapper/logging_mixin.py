import time
from .logger import logger
from .security_utils import SecurityUtils

class LoggingMixin:
    def log_call_start(self, provider: str, model_name: str, message_count: int):
        logger.info(f"Calling {provider}/{model_name} with {message_count} message(s)")
        return time.time()

    def log_call_end(self, provider: str, model_name: str, start_time: float):
        elapsed = time.time() - start_time
        logger.info(f"{provider}/{model_name} response received in {elapsed:.2f} seconds")

    def log_token_usage(self, provider: str, usage: dict):
        if usage:
            # Sanitize usage information before logging
            sanitized_usage = SecurityUtils.mask_sensitive_data(usage)
            logger.info(f"{provider} - Prompt tokens: {sanitized_usage.get('prompt_tokens')}, "
                        f"Completion tokens: {sanitized_usage.get('completion_tokens')}, "
                        f"Total: {sanitized_usage.get('total_tokens')}")
        else:
            logger.warning(f"{provider} - Token usage information not available.")
            
    def log_provider_init(self, provider: str, model: str):
        logger.info(f"Initialized {provider} wrapper with model: {model}")
        
    def log_security_event(self, event_type: str, details: dict):
        """Log security-related events with sanitized details."""
        SecurityUtils.log_security_event(event_type, details)
import logging
import os
import re
from typing import Any

LOG_LEVEL = os.getenv("LLMWRAPPER_LOG_LEVEL", "INFO").upper()

class SecureFormatter(logging.Formatter):
    """Custom formatter that masks sensitive information in log messages."""
    
    # Patterns for detecting and masking sensitive information
    SENSITIVE_PATTERNS = [
        # API Keys
        (r'sk-[a-zA-Z0-9]{32,}', 'sk-***'),           # OpenAI style
        (r'sk-ant-[a-zA-Z0-9_-]{32,}', 'sk-ant-***'), # Anthropic style
        (r'AIza[0-9A-Za-z_-]{32,}', 'AIza***'),       # Google API keys
        (r'xai-[a-zA-Z0-9]{32,}', 'xai-***'),         # xAI style
        
        # Generic patterns
        (r'api[_-]?key["\']?\s*[:=]\s*["\']?([^"\'\s,}]{8,})', r'api_key: "***"'),
        (r'token["\']?\s*[:=]\s*["\']?([^"\'\s,}]{8,})', r'token: "***"'),
        (r'password["\']?\s*[:=]\s*["\']?([^"\'\s,}]{4,})', r'password: "***"'),
        (r'secret["\']?\s*[:=]\s*["\']?([^"\'\s,}]{8,})', r'secret: "***"'),
        (r'credential["\']?\s*[:=]\s*["\']?([^"\'\s,}]{8,})', r'credential: "***"'),
        (r'auth["\']?\s*[:=]\s*["\']?([^"\'\s,}]{8,})', r'auth: "***"'),
        
        # URL with credentials
        (r'https?://[^:]+:([^@]+)@', r'https://user:***@'),
        
        # Bearer tokens
        (r'Bearer\s+([a-zA-Z0-9_-]{16,})', r'Bearer ***'),
        (r'bearer\s+([a-zA-Z0-9_-]{16,})', r'bearer ***'),
    ]
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with sensitive information masked."""
        # First, apply the standard formatting
        formatted = super().format(record)
        
        # Then mask sensitive information
        for pattern, replacement in self.SENSITIVE_PATTERNS:
            formatted = re.sub(pattern, replacement, formatted, flags=re.IGNORECASE)
        
        return formatted

# Create secure logger instance
logger = logging.getLogger("llmwrapper")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Clear any existing handlers to avoid duplicates
logger.handlers.clear()

# Create secure formatter
secure_formatter = SecureFormatter("[%(asctime)s] [%(levelname)s] %(message)s")

# Add secure stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(secure_formatter)
logger.addHandler(stream_handler)

# Optional: Add secure file handler if needed
try:
    file_handler = logging.FileHandler("llmwrapper.log")
    file_handler.setFormatter(secure_formatter)
    logger.addHandler(file_handler)
except Exception as e:
    # Use a basic message here to avoid potential recursive logging issues
    print(f"Warning: Could not create file logger: {e}")

# Prevent propagation to avoid duplicate logs
logger.propagate = False
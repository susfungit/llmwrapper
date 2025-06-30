import logging
import os

LOG_LEVEL = os.getenv("LLMWRAPPER_LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("llmwrapper")
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

stream_handler = logging.StreamHandler()
stream_format = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
stream_handler.setFormatter(stream_format)
logger.addHandler(stream_handler)

# Optional: Add file handler if needed
try:
    file_handler = logging.FileHandler("llmwrapper.log")
    file_handler.setFormatter(stream_format)
    logger.addHandler(file_handler)
except Exception as e:
    logger.warning("Could not create file logger: %s", e)
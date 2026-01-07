"""
Data Copilot Lab - Logging Configuration
Centralized logging setup with structured logging support
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from pythonjsonlogger import jsonlogger

from src.core.config import settings


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional context"""

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['app_name'] = settings.app_name
        log_record['environment'] = settings.environment
        log_record['level'] = record.levelname
        log_record['logger'] = record.name


def setup_logger(
    name: str,
    log_level: Optional[str] = None,
    log_file: Optional[Path] = None,
    use_json: bool = False
) -> logging.Logger:
    """
    Setup and configure a logger

    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path for file logging
        use_json: Whether to use JSON formatting (useful for production)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Set log level
    level = log_level or settings.log_level
    logger.setLevel(getattr(logging, level))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    # Format
    if use_json or settings.is_production:
        formatter = CustomJsonFormatter(
            '%(timestamp)s %(level)s %(name)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with default configuration

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    log_dir = Path("logs")
    log_file = log_dir / f"{settings.app_name.lower().replace(' ', '_')}.log"

    return setup_logger(
        name=name,
        log_level=settings.log_level,
        log_file=log_file,
        use_json=settings.is_production
    )


# Application logger
app_logger = get_logger(__name__)

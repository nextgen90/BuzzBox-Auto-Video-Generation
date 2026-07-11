import sys
from pathlib import Path
from loguru import logger
from app.core.config import settings

def setup_logging() -> None:
    """
    Configure loguru logger with appropriate handlers, levels, formatting,
    and automatic file rotation/compression/retention.
    """
    # Ensure the log directory exists
    log_dir = settings.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "buzzbox.log"

    # Remove all default handlers to avoid duplicate prints
    logger.remove()

    # Console output handler
    logger.add(
        sys.stdout,
        level=settings.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        backtrace=True,
        diagnose=True,
    )

    # Rotating file handler
    logger.add(
        str(log_file),
        level=settings.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    logger.info("Centralized logging has been successfully configured.")

"""
NIFTY Quant Lab - Logging Setup
=================================
Structured logging with file rotation and Telegram handler support.
"""

from __future__ import annotations

import io
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from nifty_quant_lab.config.settings import settings


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure root logger with:
    - Console handler (stdout)
    - Rotating file handler
    - Structured format with timestamps
    """
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    log_path = log_file or (settings.logs_dir / "nifty_quant_lab.log")

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if root_logger.handlers:
        return root_logger

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console — force UTF-8 so Unicode chars (✓ ✗ →) render on Windows
    console_stream = (
        io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        if hasattr(sys.stdout, "buffer")
        else sys.stdout
    )
    console_handler = logging.StreamHandler(console_stream)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Rotating file — 10 MB per file, keep 10 backups
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    file_handler.setLevel(log_level)
    root_logger.addHandler(file_handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "asyncio", "sqlalchemy.engine"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a named child logger."""
    return logging.getLogger(f"nql.{name}")

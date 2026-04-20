from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file_path: str = "logs/mcp-rag.log",
    *,
    force: bool = False,
) -> None:
    log_path = Path(log_file_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_log_path = log_path.resolve()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    if force:
        for handler in list(root_logger.handlers):
            if isinstance(handler, RotatingFileHandler):
                root_logger.removeHandler(handler)
                handler.close()
    else:
        for handler in root_logger.handlers:
            if not isinstance(handler, RotatingFileHandler):
                continue
            if Path(handler.baseFilename).resolve() == resolved_log_path:
                handler.setFormatter(formatter)
                root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
                return

    file_handler = RotatingFileHandler(
        filename=log_path,
        maxBytes=5_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    root_logger.addHandler(file_handler)

from __future__ import annotations

import logging
from pathlib import Path

from app.logging_config import setup_logging


def test_setup_logging_writes_to_file(tmp_path: Path) -> None:
    log_file = tmp_path / "mcp-rag.log"
    setup_logging(level="INFO", log_file_path=str(log_file), force=True)

    logger = logging.getLogger("tests.logging")
    logger.info("test log message")

    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "test log message" in content

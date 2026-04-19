from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.domain.interfaces import DocumentParser
from app.domain.models import Document, SourceType


class FileParser(DocumentParser):
    def __init__(self, *, folder_path: str, recursive: bool = True) -> None:
        self._folder = Path(folder_path).expanduser().resolve()
        self._recursive = recursive

    def fetch(self) -> list[dict[str, Any]]:
        iterator = self._folder.rglob("*.md") if self._recursive else self._folder.glob("*.md")
        pairs: list[dict[str, Any]] = []
        for md_path in iterator:
            if not md_path.is_file():
                continue
            json_path = md_path.with_suffix(".json")
            if not json_path.is_file():
                continue
            pairs.append({"md_path": str(md_path), "json_path": str(json_path)})
        return sorted(pairs, key=lambda item: item["md_path"])

    def parse(self, raw: dict[str, Any]) -> Document:
        md_path = Path(str(raw["md_path"]))
        json_path = Path(str(raw["json_path"]))
        text = md_path.read_text(encoding="utf-8", errors="ignore")
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        return Document(
            title=str(metadata["title"]),
            content=text,
            source=SourceType(str(metadata["source"])),
            url=str(metadata["url"]),
            created_at=self._parse_datetime(str(metadata["created_at"])),
            updated_at=self._parse_datetime(str(metadata["updated_at"])),
        )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

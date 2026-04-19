from __future__ import annotations

import sqlite3
from pathlib import Path

from app.domain.interfaces import DocumentStorage
from app.domain.models import Document


class SqliteDocumentStorage(DocumentStorage):
    def __init__(self, database_path: str = "storage/documents.db") -> None:
        db_path = Path(database_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = str(db_path)
        self._ensure_schema()

    def save(self, document: Document) -> None:
        with sqlite3.connect(self._database_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO documents (
                    id, title, content, source, url, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(document.id),
                    document.title,
                    document.content,
                    document.source.value,
                    str(document.url),
                    document.created_at.isoformat(),
                    document.updated_at.isoformat(),
                ),
            )
            conn.commit()

    def get(self, document_id: str) -> Document | None:
        with sqlite3.connect(self._database_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                """
                SELECT id, title, content, source, url, created_at, updated_at
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()
        if row is None:
            return None
        return Document(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            source=row["source"],
            url=row["url"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self._database_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source TEXT NOT NULL,
                    url TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

from __future__ import annotations

from datetime import UTC, datetime

from app.domain.models import Document, SourceType
from app.infrastructure.document_storage import SqliteDocumentStorage


def test_sqlite_document_storage_save_and_get(tmp_path) -> None:
    db_path = tmp_path / "storage.db"
    storage = SqliteDocumentStorage(database_path=str(db_path))
    document = Document(
        title="Doc",
        content="Body",
        source=SourceType.GITHUB,
        url="https://example.com/doc",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 2, tzinfo=UTC),
    )

    storage.save(document)
    loaded = storage.get(str(document.id))

    assert loaded is not None
    assert loaded.id == document.id
    assert loaded.title == "Doc"
    assert loaded.content == "Body"


def test_sqlite_document_storage_get_returns_none(tmp_path) -> None:
    storage = SqliteDocumentStorage(database_path=str(tmp_path / "storage.db"))
    assert storage.get("missing-id") is None

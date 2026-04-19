from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

import server
from app.domain.models import Chunk, Document, SourceType


class FakeEmbedder:
    def embed(self, text: str) -> list[float]:
        assert text == "hello"
        return [0.1, 0.2]


class FakeStorage:
    def __init__(self) -> None:
        self._chunk = Chunk(
            document_id=uuid4(),
            content="chunk content",
            metadata={"source": "test"},
        )
        self.last_where: dict | None = None

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float | None = None,
        where: dict | None = None,
    ) -> list[Chunk]:
        assert query_embedding == [0.1, 0.2]
        assert top_k == 5
        assert min_similarity == server._SETTINGS.SEARCH_MIN_SIMILARITY
        self.last_where = where
        return [self._chunk]

    def get(self, chunk_id: str) -> Chunk | None:
        if chunk_id == "missing":
            return None
        return self._chunk


class FakeDocumentStorage:
    def __init__(self) -> None:
        self._document = Document(
            title="Doc title",
            content="Doc body",
            source=SourceType.GITHUB,
            url="https://example.com/doc",
            created_at=datetime(2025, 1, 1, tzinfo=UTC),
            updated_at=datetime(2025, 1, 2, tzinfo=UTC),
        )

    def get(self, document_id: str) -> Document | None:
        if document_id == "missing":
            return None
        return self._document


def test_search_chunks_happy_path(monkeypatch) -> None:
    storage = FakeStorage()
    monkeypatch.setattr(server, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(server, "_get_chunk_storage", lambda: storage)

    result = server.search_chunks(query="hello", top_k=5)

    assert result["query"] == "hello"
    assert result["top_k"] == 5
    assert len(result["results"]) == 1
    assert result["results"][0]["content"] == "chunk content"
    assert storage.last_where is None


def test_search_chunks_passes_where_filter(monkeypatch) -> None:
    storage = FakeStorage()
    monkeypatch.setattr(server, "_get_embedder", lambda: FakeEmbedder())
    monkeypatch.setattr(server, "_get_chunk_storage", lambda: storage)

    result = server.search_chunks(query="hello", top_k=5, where={"document_id": "doc-1"})

    assert result["query"] == "hello"
    assert storage.last_where == {"document_id": "doc-1"}


def test_search_chunks_validates_input() -> None:
    with pytest.raises(ValidationError):
        server.search_chunks(query="", top_k=5)


def test_get_chunk_by_id_found(monkeypatch) -> None:
    storage = FakeStorage()
    monkeypatch.setattr(server, "_get_chunk_storage", lambda: storage)

    result = server.get_chunk_by_id(chunk_id="known-id")

    assert result["found"] is True
    assert result["chunk"] is not None
    assert result["chunk"]["content"] == "chunk content"


def test_get_chunk_by_id_not_found(monkeypatch) -> None:
    storage = FakeStorage()
    monkeypatch.setattr(server, "_get_chunk_storage", lambda: storage)

    result = server.get_chunk_by_id(chunk_id="missing")

    assert result["found"] is False
    assert result["chunk"] is None


def test_get_document_by_id_found(monkeypatch) -> None:
    document_storage = FakeDocumentStorage()
    monkeypatch.setattr(server, "_get_document_storage", lambda: document_storage)

    result = server.get_document_by_id(document_id="known-id")

    assert result["found"] is True
    assert result["document"] is not None
    assert result["document"]["title"] == "Doc title"


def test_get_document_by_id_not_found(monkeypatch) -> None:
    document_storage = FakeDocumentStorage()
    monkeypatch.setattr(server, "_get_document_storage", lambda: document_storage)

    result = server.get_document_by_id(document_id="missing")

    assert result["found"] is False
    assert result["document"] is None

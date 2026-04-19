from __future__ import annotations

import json
from uuid import uuid4

from app.domain.models import Chunk
from app.infrastructure.storage import ChromaStorage


class FakeCollection:
    def __init__(self) -> None:
        self.upsert_args = None
        self.query_kwargs = None
        self.get_result = {"ids": [], "documents": [], "metadatas": []}
        self.query_result = {"ids": [[]], "documents": [[]], "metadatas": [[]]}

    def upsert(self, **kwargs) -> None:
        self.upsert_args = kwargs

    def get(self, **_kwargs):
        return self.get_result

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return self.query_result


class FakeClient:
    def __init__(self, path: str) -> None:
        self.path = path
        self.collection = FakeCollection()

    def get_or_create_collection(self, name: str, metadata: dict):
        assert name == "test-collection"
        assert metadata == {"hnsw:space": "cosine"}
        return self.collection


def _make_storage(monkeypatch) -> ChromaStorage:
    holder: dict[str, FakeClient] = {}

    def factory(path: str) -> FakeClient:
        client = FakeClient(path)
        holder["client"] = client
        return client

    monkeypatch.setattr("app.infrastructure.storage.chromadb.PersistentClient", factory)
    storage = ChromaStorage(persist_directory="persist-dir", collection_name="test-collection")
    storage._fake_client = holder["client"]  # type: ignore[attr-defined]
    return storage


def test_save_writes_to_collection(monkeypatch) -> None:
    storage = _make_storage(monkeypatch)
    chunk = Chunk(
        document_id=uuid4(),
        content="text",
        metadata={
            "k": "v",
            "source_ext": ".md",
            "chunk_index": 1,
            "is_active": True,
            "extra": {"nested": 1},
        },
    )
    storage.save(chunk, [0.5, 0.6])

    args = storage._fake_client.collection.upsert_args  # type: ignore[attr-defined]
    assert args["ids"] == [str(chunk.id)]
    assert args["documents"] == ["text"]
    assert args["embeddings"] == [[0.5, 0.6]]
    metadata = args["metadatas"][0]
    assert metadata["document_id"] == str(chunk.document_id)
    assert metadata["k"] == "v"
    assert metadata["source_ext"] == ".md"
    assert metadata["chunk_index"] == 1
    assert metadata["is_active"] is True
    assert json.loads(metadata["extra"]) == {"nested": 1}


def test_get_returns_none_when_missing(monkeypatch) -> None:
    storage = _make_storage(monkeypatch)
    storage._fake_client.collection.get_result = {"ids": [], "documents": [], "metadatas": []}  # type: ignore[attr-defined]
    assert storage.get("missing") is None


def test_get_restores_chunk(monkeypatch) -> None:
    storage = _make_storage(monkeypatch)
    document_id = str(uuid4())
    chunk_id = str(uuid4())
    storage._fake_client.collection.get_result = {  # type: ignore[attr-defined]
        "ids": [chunk_id],
        "documents": ["restored"],
        "metadatas": [{"document_id": document_id, "a": 1}],
    }

    chunk = storage.get(chunk_id)

    assert chunk is not None
    assert str(chunk.id) == chunk_id
    assert str(chunk.document_id) == document_id
    assert chunk.content == "restored"
    assert chunk.metadata == {"a": 1}


def test_search_returns_chunks(monkeypatch) -> None:
    storage = _make_storage(monkeypatch)
    doc_id = str(uuid4())
    chunk_id = str(uuid4())
    storage._fake_client.collection.query_result = {  # type: ignore[attr-defined]
        "ids": [[chunk_id]],
        "documents": [["content"]],
        "metadatas": [[{"document_id": doc_id, "x": True}]],
    }

    chunks = storage.search([0.1, 0.2], top_k=3)

    assert len(chunks) == 1
    assert str(chunks[0].id) == chunk_id
    assert str(chunks[0].document_id) == doc_id
    assert chunks[0].metadata == {"x": True}


def test_search_applies_min_similarity_filter(monkeypatch) -> None:
    storage = _make_storage(monkeypatch)
    doc_id = str(uuid4())
    chunk_keep = str(uuid4())
    chunk_drop = str(uuid4())
    storage._fake_client.collection.query_result = {  # type: ignore[attr-defined]
        "ids": [[chunk_keep, chunk_drop]],
        "documents": [["good", "bad"]],
        "metadatas": [[
            {"document_id": doc_id},
            {"document_id": doc_id},
        ]],
        "distances": [[0.1, 0.7]],
    }

    chunks = storage.search([0.1, 0.2], top_k=2, min_similarity=0.5)

    assert len(chunks) == 1
    assert str(chunks[0].id) == chunk_keep


def test_search_passes_where_filter(monkeypatch) -> None:
    storage = _make_storage(monkeypatch)
    storage._fake_client.collection.query_result = {  # type: ignore[attr-defined]
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
    }

    storage.search([0.1, 0.2], top_k=3, where={"document_id": "doc-1"})

    kwargs = storage._fake_client.collection.query_kwargs  # type: ignore[attr-defined]
    assert kwargs["where"] == {"document_id": "doc-1"}


def test_parse_metadata_handles_bad_json(monkeypatch) -> None:
    storage = _make_storage(monkeypatch)
    parsed = storage._parse_metadata({"document_id": "d", "k": "v", "chunk_metadata_json": "{bad"})
    assert parsed == {"document_id": "d", "chunk_metadata": {"k": "v"}}

from __future__ import annotations

from pathlib import Path

import pytest

import loader
from app.parsers import ConfluenceParser, FileParser
from app.settings import AppSettings


class FakeEmbedder:
    def __init__(self) -> None:
        self.batch_calls: list[list[str]] = []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls.append(texts)
        return [[float(i)] for i, _ in enumerate(texts, start=1)]


class FakeStorage:
    def __init__(self) -> None:
        self.saved: list[tuple[str, list[float]]] = []

    def save(self, chunk, embedding: list[float]) -> None:
        self.saved.append((chunk.content, embedding))


class FakeDocumentStorage:
    def __init__(self) -> None:
        self.saved_ids: list[str] = []

    def save(self, document) -> None:
        self.saved_ids.append(str(document.id))


def _install_fake_splitter(monkeypatch, chunks: list[str]) -> None:
    class FakeSplitter:
        def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_text(self, _text: str) -> list[str]:
            return chunks

    monkeypatch.setattr(loader, "RecursiveCharacterTextSplitter", FakeSplitter)


def _write_pair(
    folder: Path,
    name: str,
    markdown: str,
    *,
    source: str = "github",
) -> None:
    (folder / f"{name}.md").write_text(markdown, encoding="utf-8")
    (folder / f"{name}.json").write_text(
        (
            "{\n"
            f'  "title": "{name}",\n'
            f'  "source": "{source}",\n'
            '  "url": "https://example.com/doc",\n'
            '  "created_at": "2026-04-18T10:00:00Z",\n'
            '  "updated_at": "2026-04-18T12:00:00Z"\n'
            "}\n"
        ),
        encoding="utf-8",
    )


def test_iter_document_pairs_respects_recursive_and_matching_name(tmp_path: Path) -> None:
    (tmp_path / "a.md").write_text("x", encoding="utf-8")
    (tmp_path / "a.json").write_text("{}", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.md").write_text("y", encoding="utf-8")
    (tmp_path / "sub" / "b.json").write_text("{}", encoding="utf-8")
    (tmp_path / "ignored.py").write_text("z", encoding="utf-8")
    (tmp_path / "orphan.md").write_text("z", encoding="utf-8")

    non_recursive = loader._iter_document_pairs(tmp_path, recursive=False)
    recursive = loader._iter_document_pairs(tmp_path, recursive=True)

    assert (tmp_path / "a.md", tmp_path / "a.json") in non_recursive
    assert (tmp_path / "sub" / "b.md", tmp_path / "sub" / "b.json") not in non_recursive
    assert (tmp_path / "sub" / "b.md", tmp_path / "sub" / "b.json") in recursive
    assert all(pair[0].name != "orphan.md" for pair in recursive)


def test_read_markdown_text(tmp_path: Path) -> None:
    md_path = tmp_path / "page.md"
    md_path.write_text("# Hello\nworld", encoding="utf-8")
    assert loader._read_markdown_text(md_path) == "# Hello\nworld"


def test_load_folder_to_rag_impl_indexes_and_skips(tmp_path: Path, monkeypatch) -> None:
    _write_pair(tmp_path, "doc", "some text")
    _write_pair(tmp_path, "empty", "   ")
    (tmp_path / "orphan.md").write_text("some text", encoding="utf-8")
    _install_fake_splitter(monkeypatch, ["part 1", "part 2"])

    settings = AppSettings(
        LOADER_FOLDER_PATH=str(tmp_path),
        LOADER_RECURSIVE=False,
        LOADER_CHUNK_SIZE=500,
        LOADER_CHUNK_OVERLAP=50,
    )
    embedder = FakeEmbedder()
    chunk_storage = FakeStorage()
    document_storage = FakeDocumentStorage()

    result = loader.load_folder_to_rag_impl(
        embedder=embedder,
        chunk_storage=chunk_storage,
        document_storage=document_storage,
        settings=settings,
    )

    assert result["scanned_files"] == 2
    assert result["indexed_files"] == 1
    assert result["indexed_chunks"] == 2
    assert len(result["skipped_files"]) == 1
    assert embedder.batch_calls == [["part 1", "part 2"]]
    assert [content for content, _ in chunk_storage.saved] == ["part 1", "part 2"]
    assert len(document_storage.saved_ids) == 1


def test_load_folder_to_rag_impl_raises_for_missing_folder(tmp_path: Path) -> None:
    settings = AppSettings(LOADER_FOLDER_PATH=str(tmp_path / "missing"))
    with pytest.raises(ValueError):
        loader.load_folder_to_rag_impl(
            embedder=FakeEmbedder(),
            chunk_storage=FakeStorage(),
            document_storage=FakeDocumentStorage(),
            settings=settings,
        )


def test_load_folder_to_rag_impl_skips_when_json_invalid(tmp_path: Path, monkeypatch, caplog) -> None:
    (tmp_path / "doc.md").write_text("some text", encoding="utf-8")
    (tmp_path / "doc.json").write_text("{invalid", encoding="utf-8")
    _install_fake_splitter(monkeypatch, ["part 1"])
    caplog.set_level("WARNING")

    settings = AppSettings(
        LOADER_FOLDER_PATH=str(tmp_path),
        LOADER_RECURSIVE=False,
    )
    result = loader.load_folder_to_rag_impl(
        embedder=FakeEmbedder(),
        chunk_storage=FakeStorage(),
        document_storage=FakeDocumentStorage(),
        settings=settings,
    )

    assert result["scanned_files"] == 1
    assert result["indexed_files"] == 0
    assert result["indexed_chunks"] == 0
    assert result["skipped_files"] == [str(tmp_path / "doc.md")]
    assert "reason=parse_error" in caplog.text


def test_run_loader_wires_components(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class StubEmbedder:
        def __init__(self, model_name: str, cache_folder: str | None = None) -> None:
            captured["model_name"] = model_name
            captured["cache_folder"] = cache_folder

    class StubStorage:
        def __init__(self, persist_directory: str, collection_name: str) -> None:
            captured["persist_directory"] = persist_directory
            captured["collection_name"] = collection_name

    class StubDocumentStorage:
        def __init__(self, database_path: str) -> None:
            captured["documents_database_path"] = database_path

    def fake_impl(embedder, chunk_storage, document_storage, settings):
        captured["embedder"] = embedder
        captured["chunk_storage"] = chunk_storage
        captured["document_storage"] = document_storage
        captured["settings"] = settings
        return {"ok": True}

    monkeypatch.setattr(loader, "BAAIEmbedder", StubEmbedder)
    monkeypatch.setattr(loader, "ChromaChunkStorage", StubStorage)
    monkeypatch.setattr(loader, "SqliteDocumentStorage", StubDocumentStorage)
    monkeypatch.setattr(loader, "load_folder_to_rag_impl", fake_impl)

    settings = AppSettings(
        BAAI_MODEL_NAME="my/model",
        STORAGE_DIRECTORY="my/chroma",
        CHROMA_COLLECTION="my-collection",
    )
    result = loader.run_loader(settings)

    assert result == {"ok": True}
    assert captured["model_name"] == "my/model"
    assert captured["cache_folder"] == settings.MODELS_DIRECTORY
    assert captured["persist_directory"] == settings.STORAGE_DIRECTORY
    assert captured["documents_database_path"] == settings.DOCUMENTS_DATABASE_PATH
    assert captured["collection_name"] == "my-collection"


def test_get_document_parser_file() -> None:
    settings = AppSettings(PARSER_TYPE="file")
    parser = loader._get_document_parser(settings)
    assert isinstance(parser, FileParser)


def test_get_document_parser_confluence_requires_credentials() -> None:
    settings = AppSettings(PARSER_TYPE="confluence")
    with pytest.raises(ValueError):
        loader._get_document_parser(settings)


def test_get_document_parser_confluence_uses_output_folder() -> None:
    settings = AppSettings(
        PARSER_TYPE="confluence",
        CONFLUENCE_BASE_URL="https://example.atlassian.net",
        CONFLUENCE_EMAIL="user@example.com",
        CONFLUENCE_API_TOKEN="token",
    )
    parser = loader._get_document_parser(settings)
    assert isinstance(parser, ConfluenceParser)

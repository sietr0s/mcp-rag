from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import ValidationError

from app.domain.interfaces import ChunkStorage, DocumentParser, DocumentStorage, Embedder
from app.domain.models import Chunk, Document, SourceType
from app.infrastructure.document_storage import SqliteDocumentStorage
from app.infrastructure.embedders import BAAIEmbedder
from app.infrastructure.storage import ChromaChunkStorage
from app.logging_config import setup_logging
from app.parsers import ConfluenceParser, FileParser
from app.settings import AppSettings

logger = logging.getLogger(__name__)


def _iter_document_pairs(folder: Path, recursive: bool) -> list[tuple[Path, Path]]:
    iterator = folder.rglob("*.md") if recursive else folder.glob("*.md")
    pairs: list[tuple[Path, Path]] = []
    for md_path in iterator:
        if not md_path.is_file():
            continue
        json_path = md_path.with_suffix(".json")
        if json_path.is_file():
            pairs.append((md_path, json_path))
    return sorted(pairs, key=lambda pair: str(pair[0]))


def _read_markdown_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_document_metadata(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_datetime(value: str) -> datetime:
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _build_document(md_path: Path, text: str, metadata: dict) -> Document:
    return Document(
        title=str(metadata["title"]),
        content=text,
        source=SourceType(str(metadata["source"])),
        url=str(metadata["url"]),
        created_at=_parse_datetime(str(metadata["created_at"])),
        updated_at=_parse_datetime(str(metadata["updated_at"])),
    )


def _get_document_parser(settings: AppSettings) -> DocumentParser:
    parser_type = settings.PARSER_TYPE.strip().lower()
    if parser_type == "file":
        folder = Path(settings.LOADER_FOLDER_PATH).expanduser().resolve()
        if not folder.exists() or not folder.is_dir():
            raise ValueError(f"Folder does not exist or is not a directory: {folder}")
        return FileParser(
            folder_path=settings.LOADER_FOLDER_PATH,
            recursive=settings.LOADER_RECURSIVE,
        )
    if parser_type == "confluence":
        if not settings.CONFLUENCE_BASE_URL or not settings.CONFLUENCE_EMAIL or not settings.CONFLUENCE_API_TOKEN:
            raise ValueError("CONFLUENCE_BASE_URL, CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN are required")
        return ConfluenceParser(
            base_url=settings.CONFLUENCE_BASE_URL,
            email=settings.CONFLUENCE_EMAIL,
            api_token=settings.CONFLUENCE_API_TOKEN,
            space_key=settings.CONFLUENCE_SPACE_KEY,
            limit=settings.CONFLUENCE_LIMIT,
            output_folder=settings.LOADER_FOLDER_PATH,
        )
    raise ValueError(f"Unsupported PARSER_TYPE: {settings.PARSER_TYPE}")


def load_folder_to_rag_impl(
    embedder: Embedder,
    chunk_storage: ChunkStorage,
    document_storage: DocumentStorage,
    settings: AppSettings,
    parser: DocumentParser | None = None,
) -> dict:
    setup_logging(level=settings.LOG_LEVEL, log_file_path=settings.LOG_FILE_PATH)
    source_dir = Path(settings.LOADER_FOLDER_PATH).expanduser().resolve()
    logger.info("Starting load_folder_to_rag parser_type=%s", settings.PARSER_TYPE)
    parser = parser or _get_document_parser(settings)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.LOADER_CHUNK_SIZE,
        chunk_overlap=settings.LOADER_CHUNK_OVERLAP,
    )

    raw_documents = parser.fetch()
    logger.info("Fetched raw documents=%d", len(raw_documents))
    indexed_files = 0
    indexed_chunks = 0
    skipped_files: list[str] = []

    for raw in raw_documents:
        source_key = str(raw.get("md_path") or raw.get("id") or raw.get("title") or "<unknown>")
        try:
            document = parser.parse(raw)
        except (json.JSONDecodeError, KeyError, ValidationError, ValueError) as exc:
            logger.warning(
                "Skipping document source=%s reason=parse_error error=%s",
                source_key,
                exc,
            )
            skipped_files.append(source_key)
            continue

        text = document.content
        if not text.strip():
            logger.warning("Skipping document source=%s reason=empty_content", source_key)
            skipped_files.append(source_key)
            continue

        parts = [part.strip() for part in splitter.split_text(text) if part.strip()]
        if not parts:
            logger.warning("Skipping document source=%s reason=no_chunks_after_split", source_key)
            skipped_files.append(source_key)
            continue

        document_storage.save(document)

        source_path = str(raw.get("md_path", source_key))
        source_name = Path(source_path).name if raw.get("md_path") else document.title
        source_ext = Path(source_path).suffix.lower() if raw.get("md_path") else ""

        chunks = [
            Chunk(
                document_id=document.id,
                content=part,
                metadata={
                    "document_id": str(document.id),
                    "source_path": source_path,
                    "source_name": source_name,
                    "source_ext": source_ext,
                    "source": document.source.value,
                    "chunk_index": idx,
                    "chunk_count": len(parts),
                },
            )
            for idx, part in enumerate(parts)
        ]

        vectors = embedder.embed_batch([chunk.content for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            chunk_storage.save(chunk, vector)

        indexed_files += 1
        indexed_chunks += len(chunks)

    return {
        "folder_path": str(source_dir),
        "scanned_files": len(raw_documents),
        "indexed_files": indexed_files,
        "indexed_chunks": indexed_chunks,
        "skipped_files": skipped_files,
    }


def run_loader(settings: AppSettings) -> dict:
    setup_logging(level=settings.LOG_LEVEL, log_file_path=settings.LOG_FILE_PATH)
    logger.info("run_loader started")
    embedder = BAAIEmbedder(
        model_name=settings.BAAI_MODEL_NAME,
        cache_folder=settings.MODELS_DIRECTORY,
    )
    chunk_storage = ChromaChunkStorage(
        persist_directory=settings.STORAGE_DIRECTORY,
        collection_name=settings.CHROMA_COLLECTION,
    )
    document_storage = SqliteDocumentStorage(database_path=settings.DOCUMENTS_DATABASE_PATH)
    result = load_folder_to_rag_impl(
        embedder=embedder,
        chunk_storage=chunk_storage,
        document_storage=document_storage,
        settings=settings,
    )
    if "indexed_files" in result and "indexed_chunks" in result:
        logger.info(
            "run_loader completed indexed_files=%s indexed_chunks=%s",
            result["indexed_files"],
            result["indexed_chunks"],
        )
    else:
        logger.info("run_loader completed")
    return result


if __name__ == "__main__":
    result = run_loader(AppSettings())
    print(json.dumps(result, ensure_ascii=False))

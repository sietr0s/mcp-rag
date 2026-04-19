from __future__ import annotations

import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from app.domain.models import Chunk, Document
from app.infrastructure.document_storage import SqliteDocumentStorage
from app.infrastructure.embedders import BAAIEmbedder
from app.infrastructure.storage import ChromaChunkStorage
from app.logging_config import setup_logging
from app.schemas import (
    ChunkPayload,
    DocumentPayload,
    GetDocumentByIdRequest,
    GetDocumentByIdResponse,
    GetChunkByIdRequest,
    GetChunkByIdResponse,
    SearchChunksRequest,
    SearchChunksResponse,
)
from app.settings import get_settings


_SETTINGS = get_settings()
setup_logging(level=_SETTINGS.LOG_LEVEL, log_file_path=_SETTINGS.LOG_FILE_PATH)
logger = logging.getLogger(__name__)
mcp = FastMCP("mcp-rag-search")

_EMBEDDER: BAAIEmbedder | None = None
_CHUNK_STORAGE: ChromaChunkStorage | None = None
_DOCUMENT_STORAGE: SqliteDocumentStorage | None = None


def _get_embedder() -> BAAIEmbedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        logger.info("Initializing embedder model=%s", _SETTINGS.BAAI_MODEL_NAME)
        _EMBEDDER = BAAIEmbedder(
            model_name=_SETTINGS.BAAI_MODEL_NAME,
            cache_folder=_SETTINGS.MODELS_DIRECTORY,
        )
    return _EMBEDDER


def _get_chunk_storage() -> ChromaChunkStorage:
    global _CHUNK_STORAGE
    if _CHUNK_STORAGE is None:
        logger.info(
            "Initializing storage dir=%s collection=%s",
            _SETTINGS.STORAGE_DIRECTORY,
            _SETTINGS.CHROMA_COLLECTION,
        )
        _CHUNK_STORAGE = ChromaChunkStorage(
            persist_directory=_SETTINGS.STORAGE_DIRECTORY,
            collection_name=_SETTINGS.CHROMA_COLLECTION,
        )
    return _CHUNK_STORAGE


def _get_document_storage() -> SqliteDocumentStorage:
    global _DOCUMENT_STORAGE
    if _DOCUMENT_STORAGE is None:
        logger.info("Initializing document storage db=%s", _SETTINGS.DOCUMENTS_DATABASE_PATH)
        _DOCUMENT_STORAGE = SqliteDocumentStorage(database_path=_SETTINGS.DOCUMENTS_DATABASE_PATH)
    return _DOCUMENT_STORAGE


def _initialize_runtime() -> None:
    logger.info(
        "Initializing runtime dependencies before MCP server start parser_type=%s",
        _SETTINGS.PARSER_TYPE,
    )
    _get_embedder()
    _get_chunk_storage()
    _get_document_storage()
    logger.info("Runtime dependencies are ready")


def _to_payload(chunk: Chunk) -> ChunkPayload:
    return ChunkPayload(
        id=str(chunk.id),
        document_id=str(chunk.document_id),
        content=chunk.content,
        metadata=chunk.metadata,
    )


def _to_document_payload(document: Document) -> DocumentPayload:
    return DocumentPayload(
        id=str(document.id),
        title=document.title,
        content=document.content,
        source=document.source.value,
        url=str(document.url),
        created_at=document.created_at.isoformat(),
        updated_at=document.updated_at.isoformat(),
    )


@mcp.tool(
    description=(
        "Semantic search over indexed chunks by query text. "
        "Use optional `where` to filter by metadata fields before similarity ranking. "
        "Supported fields include `document_id` and chunk metadata keys such as "
        "`source_path`, `source_name`, `source_ext`, `chunk_index`, `chunk_count`. "
        "Example: `where={\"document_id\": \"...\", \"source_ext\": \".md\"}`."
    )
)
def search_chunks(query: str, top_k: int = 5, where: dict[str, Any] | None = None) -> dict:
    """Return top-k relevant chunks for the input query with optional metadata filter."""
    logger.info("search_chunks query=%r top_k=%d where=%s", query, top_k, where)
    request = SearchChunksRequest(query=query, top_k=top_k, where=where)
    query_embedding = _get_embedder().embed(request.query)
    chunks = _get_chunk_storage().search(
        query_embedding=query_embedding,
        top_k=request.top_k,
        min_similarity=_SETTINGS.SEARCH_MIN_SIMILARITY,
        where=request.where,
    )
    response = SearchChunksResponse(
        query=request.query,
        top_k=request.top_k,
        results=[_to_payload(chunk) for chunk in chunks],
    )
    logger.info("search_chunks done results=%d", len(response.results))
    return response.model_dump()


@mcp.tool(description="Get a chunk by its identifier.")
def get_chunk_by_id(chunk_id: str) -> dict:
    """Return a single chunk by id or indicate that it was not found."""
    logger.info("get_chunk_by_id chunk_id=%s", chunk_id)
    request = GetChunkByIdRequest(chunk_id=chunk_id)
    chunk = _get_chunk_storage().get(request.chunk_id)
    response = GetChunkByIdResponse(
        found=chunk is not None,
        chunk=_to_payload(chunk) if chunk else None,
    )
    logger.info("get_chunk_by_id found=%s", response.found)
    return response.model_dump()


@mcp.tool(description="Get a document by its identifier.")
def get_document_by_id(document_id: str) -> dict:
    logger.info("get_document_by_id document_id=%s", document_id)
    request = GetDocumentByIdRequest(document_id=document_id)
    document = _get_document_storage().get(request.document_id)
    response = GetDocumentByIdResponse(
        found=document is not None,
        document=_to_document_payload(document) if document else None,
    )
    logger.info("get_document_by_id found=%s", response.found)
    return response.model_dump()


if __name__ == "__main__":
    _initialize_runtime()
    mcp.run(transport="streamable-http")

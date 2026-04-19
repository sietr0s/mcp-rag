from __future__ import annotations

import json
import logging
from typing import Any

import chromadb

from app.domain.interfaces import ChunkStorage
from app.domain.models import Chunk

logger = logging.getLogger(__name__)


class ChromaChunkStorage(ChunkStorage):
    """
    ChromaDB implementation of the domain ChunkStorage interface.
    """

    def __init__(
        self,
        persist_directory: str = "storage/chroma",
        collection_name: str = "chunks",
    ) -> None:
        logger.info(
            "ChromaStorage init persist_directory=%s collection=%s",
            persist_directory,
            collection_name,
        )
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def save(self, chunk: Chunk, embedding: list[float]) -> None:
        logger.debug("ChromaStorage save chunk_id=%s", chunk.id)
        metadata = self._to_chroma_metadata(chunk)
        self._collection.upsert(
            ids=[str(chunk.id)],
            embeddings=[embedding],
            documents=[chunk.content],
            metadatas=[metadata],
        )

    def get(self, chunk_id: str) -> Chunk | None:
        logger.debug("ChromaStorage get chunk_id=%s", chunk_id)
        result = self._collection.get(
            ids=[chunk_id],
            include=["documents", "metadatas"],
        )
        if not result["ids"]:
            return None

        metadata = self._parse_metadata(result["metadatas"][0] if result["metadatas"] else None)
        content = result["documents"][0] if result["documents"] else ""

        return Chunk(
            id=chunk_id,
            document_id=metadata.get("document_id", ""),
            content=content,
            metadata=metadata.get("chunk_metadata", {}),
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        logger.debug(
            "ChromaStorage search top_k=%d min_similarity=%s where=%s",
            top_k,
            min_similarity,
            where,
        )
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[Chunk] = []
        for idx, chunk_id in enumerate(ids):
            distance = distances[idx] if idx < len(distances) else None
            if min_similarity is not None and distance is not None:
                similarity = 1.0 - float(distance)
                if similarity < min_similarity:
                    continue
            metadata = self._parse_metadata(metadatas[idx] if idx < len(metadatas) else None)
            content = documents[idx] if idx < len(documents) else ""
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=metadata.get("document_id", ""),
                    content=content,
                    metadata=metadata.get("chunk_metadata", {}),
                )
            )
        return chunks

    def _parse_metadata(self, metadata: dict[str, Any] | None) -> dict[str, Any]:
        if not metadata:
            return {}

        chunk_metadata = {
            key: value
            for key, value in metadata.items()
            if key not in {"document_id", "chunk_metadata_json"}
        }

        raw_chunk_metadata = metadata.get("chunk_metadata_json")
        try:
            legacy_metadata = json.loads(raw_chunk_metadata) if isinstance(raw_chunk_metadata, str) else {}
        except json.JSONDecodeError:
            legacy_metadata = {}

        if isinstance(legacy_metadata, dict):
            for key, value in legacy_metadata.items():
                chunk_metadata.setdefault(key, value)

        return {
            "document_id": metadata.get("document_id", ""),
            "chunk_metadata": chunk_metadata,
        }

    def _to_chroma_metadata(self, chunk: Chunk) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {}
        for key, value in chunk.metadata.items():
            metadata[str(key)] = self._to_chroma_metadata_value(value)
        metadata["document_id"] = str(chunk.document_id)
        return metadata

    @staticmethod
    def _to_chroma_metadata_value(value: Any) -> str | int | float | bool:
        if isinstance(value, bool | int | float | str):
            return value
        return json.dumps(value, ensure_ascii=False)


# Backward compatibility alias; prefer ChromaChunkStorage.
ChromaStorage = ChromaChunkStorage

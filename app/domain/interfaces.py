from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.domain.models import Chunk, Document


class DocumentParser(ABC):
    @abstractmethod
    def fetch(self) -> list[dict[str, Any]]:
        """Fetch raw documents from a source."""
        raise NotImplementedError

    @abstractmethod
    def parse(self, raw: dict[str, Any]) -> Document:
        """Convert raw source data into a domain Document."""
        raise NotImplementedError


class DocumentRepository(ABC):
    @abstractmethod
    def save(self, document: Document) -> None:
        """Persist a normalized document."""
        raise NotImplementedError

    @abstractmethod
    def get(self, document_id: str) -> Document:
        """Load a document by id."""
        raise NotImplementedError


class Embedder(ABC):
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Generate an embedding vector for a single text input."""
        raise NotImplementedError

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embedding vectors for multiple text inputs."""
        raise NotImplementedError


class ChunkStorage(ABC):
    @abstractmethod
    def save(self, chunk: Chunk, embedding: list[float]) -> None:
        """Persist embedding for a chunk."""
        raise NotImplementedError

    @abstractmethod
    def get(self, chunk_id: str) -> Chunk | None:
        """Load chunk by id."""
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        min_similarity: float | None = None,
        where: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """Return top-k most similar chunks."""
        raise NotImplementedError


class DocumentStorage(ABC):
    @abstractmethod
    def save(self, document: Document) -> None:
        """Persist a document."""
        raise NotImplementedError

    @abstractmethod
    def get(self, document_id: str) -> Document | None:
        """Load a document by id."""
        raise NotImplementedError

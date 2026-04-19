from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ChunkPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    document_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    content: str
    source: str
    url: str
    created_at: str
    updated_at: str


class SearchChunksRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    where: dict[str, Any] | None = None


class SearchChunksResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    top_k: int
    results: list[ChunkPayload] = Field(default_factory=list)


class GetChunkByIdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)


class GetChunkByIdResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    chunk: ChunkPayload | None = None


class GetDocumentByIdRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)


class GetDocumentByIdResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    found: bool
    document: DocumentPayload | None = None

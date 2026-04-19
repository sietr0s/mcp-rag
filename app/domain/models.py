from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class SourceType(str, Enum):
    LOCAL = "local"
    CONFLUENCE = "confluence"
    GITHUB = "github"
    BITBUCKET = "bitbucket"


class Chunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    document_id: UUID
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: UUID = Field(default_factory=uuid4)
    title: str
    content: str
    source: SourceType
    url: HttpUrl
    created_at: datetime
    updated_at: datetime
    chunks: list[Chunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_timestamps(self) -> "Document":
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must be greater than or equal to created_at")
        return self

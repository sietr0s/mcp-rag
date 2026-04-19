from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.domain.models import Chunk, Document, SourceType


def test_document_timestamp_validation() -> None:
    with pytest.raises(ValidationError):
        Document(
            title="t",
            content="c",
            source=SourceType.GITHUB,
            url="https://example.com",
            created_at=datetime(2025, 1, 2, tzinfo=UTC),
            updated_at=datetime(2025, 1, 1, tzinfo=UTC),
        )


def test_document_valid_minimal_payload() -> None:
    doc = Document(
        title="t",
        content="c",
        source=SourceType.CONFLUENCE,
        url="https://example.com",
        created_at=datetime(2025, 1, 1, tzinfo=UTC),
        updated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )
    assert doc.chunks == []


def test_chunk_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Chunk(
            document_id=uuid4(),
            content="c",
            metadata={},
            unexpected="x",
        )

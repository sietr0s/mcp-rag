from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import GetChunkByIdRequest, SearchChunksRequest


def test_search_chunks_request_defaults() -> None:
    request = SearchChunksRequest(query="hello")
    assert request.top_k == 5


@pytest.mark.parametrize("top_k", [0, 51])
def test_search_chunks_request_top_k_bounds(top_k: int) -> None:
    with pytest.raises(ValidationError):
        SearchChunksRequest(query="hello", top_k=top_k)


def test_search_chunks_request_query_required() -> None:
    with pytest.raises(ValidationError):
        SearchChunksRequest(query="")


def test_get_chunk_by_id_request_chunk_id_required() -> None:
    with pytest.raises(ValidationError):
        GetChunkByIdRequest(chunk_id="")

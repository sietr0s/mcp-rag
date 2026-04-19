from __future__ import annotations

import logging

from langchain_huggingface import HuggingFaceEmbeddings

from app.domain.interfaces import Embedder

logger = logging.getLogger(__name__)


class BAAIEmbedder(Embedder):
    """
    Embedder based on BAAI models via LangChain HuggingFace integration.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        cache_folder: str | None = None,
        model_kwargs: dict | None = None,
        encode_kwargs: dict | None = None,
    ) -> None:
        logger.info("BAAIEmbedder init model=%s cache_folder=%s", model_name, cache_folder)
        self._embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=cache_folder,
            model_kwargs=model_kwargs or {"device": "cpu"},
            encode_kwargs=encode_kwargs or {"normalize_embeddings": True},
        )

    def embed(self, text: str) -> list[float]:
        return list(self._embeddings.embed_query(text))

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [list(vector) for vector in self._embeddings.embed_documents(texts)]

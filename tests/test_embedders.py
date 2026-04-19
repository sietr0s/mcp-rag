from __future__ import annotations

import app.infrastructure.embedders as embedders_module
from app.infrastructure.embedders import BAAIEmbedder


def test_baai_embedder_init_and_methods(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeHFEmbeddings:
        def __init__(
            self,
            model_name: str,
            cache_folder: str | None,
            model_kwargs: dict,
            encode_kwargs: dict,
        ) -> None:
            captured["model_name"] = model_name
            captured["cache_folder"] = cache_folder
            captured["model_kwargs"] = model_kwargs
            captured["encode_kwargs"] = encode_kwargs

        def embed_query(self, text: str):
            assert text == "hello"
            return (0.1, 0.2)

        def embed_documents(self, texts: list[str]):
            assert texts == ["a", "b"]
            return [(1.0, 2.0), (3.0, 4.0)]

    monkeypatch.setattr(embedders_module, "HuggingFaceEmbeddings", FakeHFEmbeddings)

    embedder = BAAIEmbedder(model_name="custom/model")

    assert captured["model_name"] == "custom/model"
    assert captured["cache_folder"] is None
    assert captured["model_kwargs"] == {"device": "cpu"}
    assert captured["encode_kwargs"] == {"normalize_embeddings": True}
    assert embedder.embed("hello") == [0.1, 0.2]
    assert embedder.embed_batch(["a", "b"]) == [[1.0, 2.0], [3.0, 4.0]]

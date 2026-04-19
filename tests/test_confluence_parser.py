from __future__ import annotations

import json

from app.domain.models import SourceType
from app.parsers.confluence import ConfluenceParser


def test_parse_converts_confluence_payload_to_document() -> None:
    parser = ConfluenceParser(
        base_url="https://example.atlassian.net/",
        email="user@example.com",
        api_token="token",
    )
    raw = {
        "id": "12345",
        "title": "Confluence Page",
        "_links": {"webui": "/wiki/spaces/ENG/pages/12345/Confluence+Page"},
        "body": {"storage": {"value": "<h1>Hello</h1><p>world</p>"}},
        "history": {"createdDate": "2026-04-18T10:00:00.000Z"},
        "version": {"when": "2026-04-18T12:00:00.000Z"},
    }

    document = parser.parse(raw)

    assert document.title == "Confluence Page"
    assert document.content == "Hello world"
    assert document.source == SourceType.CONFLUENCE
    assert str(document.url) == "https://example.atlassian.net/wiki/spaces/ENG/pages/12345/Confluence+Page"
    assert document.created_at.isoformat() == "2026-04-18T10:00:00+00:00"
    assert document.updated_at.isoformat() == "2026-04-18T12:00:00+00:00"


def test_fetch_uses_pagination(monkeypatch) -> None:
    parser = ConfluenceParser(
        base_url="https://example.atlassian.net/",
        email="user@example.com",
        api_token="token",
        limit=2,
    )
    calls: list[str] = []

    def fake_request_json(endpoint: str):
        calls.append(endpoint)
        if "start=0" in endpoint:
            return {"results": [{"id": "1"}, {"id": "2"}]}
        return {"results": [{"id": "3"}]}

    monkeypatch.setattr(parser, "_request_json", fake_request_json)

    rows = parser.fetch()

    assert [row["id"] for row in rows] == ["1", "2", "3"]
    assert len(calls) == 2


def test_parse_persists_md_and_json_when_output_folder(tmp_path) -> None:
    parser = ConfluenceParser(
        base_url="https://example.atlassian.net/",
        email="user@example.com",
        api_token="token",
        output_folder=str(tmp_path),
    )
    raw = {
        "id": "12345",
        "title": "Confluence Page",
        "_links": {"webui": "/wiki/spaces/ENG/pages/12345/Confluence+Page"},
        "body": {"storage": {"value": "<p>hello</p>"}},
        "history": {"createdDate": "2026-04-18T10:00:00.000Z"},
        "version": {"when": "2026-04-18T12:00:00.000Z"},
    }

    parser.parse(raw)

    md_path = tmp_path / "Confluence-Page-12345.md"
    json_path = tmp_path / "Confluence-Page-12345.json"
    assert md_path.exists()
    assert json_path.exists()
    assert md_path.read_text(encoding="utf-8") == "hello"
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["confluence_page_id"] == "12345"

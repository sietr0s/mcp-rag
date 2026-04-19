from __future__ import annotations

import base64
import json
import re
from datetime import UTC, datetime
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

from app.domain.interfaces import DocumentParser
from app.domain.models import Document, SourceType


class ConfluenceParser(DocumentParser):
    def __init__(
        self,
        *,
        base_url: str,
        email: str,
        api_token: str,
        space_key: str | None = None,
        limit: int = 50,
        output_folder: str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._email = email
        self._api_token = api_token
        self._space_key = space_key
        self._limit = limit
        self._output_folder = Path(output_folder).expanduser().resolve() if output_folder else None
        if self._output_folder is not None:
            self._output_folder.mkdir(parents=True, exist_ok=True)

    def fetch(self) -> list[dict[str, Any]]:
        start = 0
        collected: list[dict[str, Any]] = []
        while True:
            query = {
                "type": "page",
                "status": "current",
                "expand": "body.storage,version,history",
                "limit": self._limit,
                "start": start,
            }
            if self._space_key:
                query["spaceKey"] = self._space_key
            endpoint = f"wiki/rest/api/content?{urlencode(query)}"
            payload = self._request_json(endpoint)
            items = payload.get("results", [])
            if not isinstance(items, list):
                break
            collected.extend(item for item in items if isinstance(item, dict))
            if len(items) < self._limit:
                break
            start += self._limit
        return collected

    def parse(self, raw: dict[str, Any]) -> Document:
        title = str(raw.get("title", "")).strip()
        page_id = str(raw.get("id", "")).strip()
        links = raw.get("_links", {}) if isinstance(raw.get("_links"), dict) else {}
        webui = str(links.get("webui", "")).strip()
        url = webui if webui.startswith("http") else urljoin(self._base_url, webui.lstrip("/"))

        body = raw.get("body", {}) if isinstance(raw.get("body"), dict) else {}
        storage = body.get("storage", {}) if isinstance(body.get("storage"), dict) else {}
        value = str(storage.get("value", ""))
        content = self._html_to_text(value)

        version = raw.get("version", {}) if isinstance(raw.get("version"), dict) else {}
        history = raw.get("history", {}) if isinstance(raw.get("history"), dict) else {}
        updated_raw = str(version.get("when", "")).strip()
        created_raw = str(history.get("createdDate", "")).strip() or updated_raw
        updated_at = self._parse_datetime(updated_raw)
        created_at = self._parse_datetime(created_raw)

        document = Document(
            title=title or f"Confluence page {page_id}" if page_id else "Confluence page",
            content=content,
            source=SourceType.CONFLUENCE,
            url=url,
            created_at=created_at,
            updated_at=updated_at,
            chunks=[],
        )
        self._persist_document_files(document, page_id=page_id)
        return document

    def _request_json(self, endpoint: str) -> dict[str, Any]:
        url = urljoin(self._base_url, endpoint.lstrip("/"))
        auth_pair = f"{self._email}:{self._api_token}".encode("utf-8")
        auth_header = base64.b64encode(auth_pair).decode("ascii")
        request = Request(
            url=url,
            method="GET",
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {auth_header}",
            },
        )
        with urlopen(request, timeout=30) as response:
            payload = response.read().decode("utf-8")
        parsed = json.loads(payload)
        return parsed if isinstance(parsed, dict) else {}

    def _persist_document_files(self, document: Document, *, page_id: str) -> None:
        if self._output_folder is None:
            return
        safe_title = self._slugify(document.title)
        base_name = f"{safe_title}-{page_id}" if page_id else safe_title
        md_path = self._output_folder / f"{base_name}.md"
        json_path = self._output_folder / f"{base_name}.json"

        md_path.write_text(document.content, encoding="utf-8")
        json_path.write_text(
            json.dumps(
                {
                    "title": document.title,
                    "source": document.source.value,
                    "url": str(document.url),
                    "created_at": document.created_at.isoformat(),
                    "updated_at": document.updated_at.isoformat(),
                    "confluence_page_id": page_id,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _html_to_text(html: str) -> str:
        text = re.sub(r"<[^>]+>", " ", html)
        text = unescape(text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _slugify(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-")
        return slug or "confluence-page"

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed

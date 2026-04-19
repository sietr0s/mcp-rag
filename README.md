# mcp-rag

MCP сервер для RAG-поиска по чанкам документов + хранение документов в SQLite.

## Что внутри

- `ChunkStorage`: `ChromaChunkStorage` (векторный поиск по чанкам).
- `DocumentStorage`: `SqliteDocumentStorage` (получение документа по `id` без векторного поиска).
- Инструменты MCP:
  - `search_chunks`
  - `get_chunk_by_id`
  - `get_document_by_id`

## Быстрый старт

### 1. Установка

```bash
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 2. Конфиг

```bash
copy .env.example .env
```

### 3. Индексация документов

```bash
.\.venv\Scripts\python.exe loader.py
```

Режим парсера задается через `PARSER_TYPE`:

- `file`: читает пары `*.md + *.json` из `LOADER_FOLDER_PATH`.
- `confluence`: тянет страницы из Confluence и сохраняет выгрузку в `LOADER_FOLDER_PATH` в формате `*.md + *.json`.

### 4. Запуск MCP сервера

```bash
.\.venv\Scripts\python.exe server.py
```

По умолчанию сервер поднимается в `streamable-http`.

## Подключение MCP клиента

```json
{
  "mcpServers": {
    "mcp-rag-search": {
      "url": "http://127.0.0.1:8000/mcp"
    }
  }
}
```

## Тесты и линт

```bash
pytest -q -k "not e2e"
ruff check .
```

E2E тесты включаются флагами окружения:

- `RUN_MCP_E2E=1`
- `RUN_SLOW_MCP_E2E=1`

## Что не нужно коммитить

См. `.gitignore`: локальные модели, хранилища данных, логи, кэш и `.env` исключены.

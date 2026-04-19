from __future__ import annotations

from pathlib import Path

from app.settings import AppSettings, get_settings


def test_app_settings_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv("BAAI_MODEL_NAME", "test/model")
    settings = AppSettings()
    assert settings.BAAI_MODEL_NAME == "test/model"


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    first = get_settings()
    second = get_settings()
    assert first is second
    get_settings.cache_clear()


def test_search_min_similarity_bounds() -> None:
    settings = AppSettings(SEARCH_MIN_SIMILARITY=0.6)
    assert settings.SEARCH_MIN_SIMILARITY == 0.6


def test_paths_are_resolved_from_project_root() -> None:
    project_root = Path(__file__).resolve().parents[1]
    settings = AppSettings(
        MODELS_DIRECTORY="models",
        STORAGE_DIRECTORY="storage/chroma",
        DOCUMENTS_DATABASE_PATH="storage/storage.db",
        LOG_FILE_PATH="logs/mcp-rag.log",
    )
    assert settings.MODELS_DIRECTORY == str((project_root / "models").resolve())
    assert settings.STORAGE_DIRECTORY == str((project_root / "storage/chroma").resolve())
    assert settings.DOCUMENTS_DATABASE_PATH == str((project_root / "storage/storage.db").resolve())
    assert settings.LOG_FILE_PATH == str((project_root / "logs/mcp-rag.log").resolve())

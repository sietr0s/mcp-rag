from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BAAI_MODEL_NAME: str = "BAAI/bge-m3"
    MODELS_DIRECTORY: str = "models"
    STORAGE_DIRECTORY: str = "storage/chroma"
    DOCUMENTS_DATABASE_PATH: str = "storage/storage.db"
    CHROMA_COLLECTION: str = "chunks"
    SEARCH_MIN_SIMILARITY: float = Field(default=0.2, ge=0.0, le=1.0)

    LOG_LEVEL: str = "DEBUG"
    LOG_FILE_PATH: str = "logs/mcp-rag.log"

    LOADER_FOLDER_PATH: str = "docs"
    LOADER_RECURSIVE: bool = True
    LOADER_CHUNK_SIZE: int = Field(default=1000, ge=100, le=8000)
    LOADER_CHUNK_OVERLAP: int = Field(default=150, ge=0, le=2000)
    PARSER_TYPE: str = "file"

    CONFLUENCE_BASE_URL: str = ""
    CONFLUENCE_EMAIL: str = ""
    CONFLUENCE_API_TOKEN: str = ""
    CONFLUENCE_SPACE_KEY: str = ""
    CONFLUENCE_LIMIT: int = Field(default=50, ge=1, le=200)

    @model_validator(mode="after")
    def normalize_paths(self) -> "AppSettings":
        self.MODELS_DIRECTORY = self._to_project_absolute_path(self.MODELS_DIRECTORY)
        self.STORAGE_DIRECTORY = self._to_project_absolute_path(self.STORAGE_DIRECTORY)
        self.DOCUMENTS_DATABASE_PATH = self._to_project_absolute_path(self.DOCUMENTS_DATABASE_PATH)
        self.LOG_FILE_PATH = self._to_project_absolute_path(self.LOG_FILE_PATH)
        return self

    @staticmethod
    def _to_project_absolute_path(path_value: str) -> str:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = _PROJECT_ROOT / path
        return str(path.resolve())


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()

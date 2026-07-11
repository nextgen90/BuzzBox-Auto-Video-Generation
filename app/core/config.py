import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """
    App-wide settings loaded from environment variables and dotenv file.
    Uses Pydantic validation for all properties.
    """
    # Application Config
    APP_NAME: str = "BuzzBox AI Engine"
    APP_VERSION: str = "1.0.0"

    # AWS S3 Storage Config (To be used in future sprints)
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET: Optional[str] = None

    # Configurable AI Model Settings
    VIDEO_ENGINE: str = "wan22"
    MODEL_NAME: str = "Wan-Video/Wan2.1-T2V-1.3B"
    MODEL_CACHE: Optional[str] = None
    MODEL_PROVIDER: str = "huggingface"
    MODEL_PRECISION: str = "float16"
    DEVICE: str = "cuda"

    # Logging
    LOG_LEVEL: str = "INFO"

    # Directory Paths (Using Pathlib for robust OS-agnostic path operations)
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    @property
    def LOG_DIR(self) -> Path:
        return self.BASE_DIR / "logs"

    @property
    def UPLOAD_DIR(self) -> Path:
        return self.BASE_DIR / "uploads"

    @property
    def OUTPUT_DIR(self) -> Path:
        return self.BASE_DIR / "outputs"

    @property
    def STORAGE_DIR(self) -> Path:
        return self.BASE_DIR / "storage"

    @property
    def CHARACTERS_DIR(self) -> Path:
        return self.STORAGE_DIR / "characters"

    @property
    def JOBS_DIR(self) -> Path:
        return self.STORAGE_DIR / "jobs"

    @property
    def PROMPTS_DIR(self) -> Path:
        return self.STORAGE_DIR / "prompts"

    @property
    def METADATA_DIR(self) -> Path:
        return self.STORAGE_DIR / "metadata"

    @property
    def CACHE_DIR(self) -> Path:
        return self.STORAGE_DIR / "cache"

    @property
    def EXPORTS_DIR(self) -> Path:
        return self.STORAGE_DIR / "exports"

    @property
    def RENDERS_DIR(self) -> Path:
        return self.STORAGE_DIR / "renders"

    # Model configuration for pydantic-settings
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

settings = Settings()

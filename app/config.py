"""Central configuration, loaded from environment / .env."""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Security
    secret_key: str = "dev-insecure-change-me"
    cookie_secure: bool = False
    cookie_name: str = "translate_session"
    session_max_age: int = 60 * 60 * 24 * 14  # 14 days

    # External services
    hf_token: str = ""
    redis_url: str = "redis://localhost:6379"

    # Storage
    data_dir: Path = Path("./data")

    # Whisper / GPU
    whisper_model: str = "large-v3"
    whisper_compute_type: str = "float16"
    whisper_device: str = "cuda"
    whisper_batch_size: int = 16

    # Speaker identification: cosine similarity above which a diarized voice is
    # auto-labeled with a known person. Tune per diarization model (0.4–0.6).
    speaker_match_threshold: float = 0.5

    # Uploads
    max_upload_mb: int = 2048
    allowed_extensions: tuple[str, ...] = (
        ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".opus",
        ".aac", ".wma", ".mp4", ".mkv", ".mov", ".webm",
    )

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def output_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def database_url(self) -> str:
        return f"sqlite:///{self.db_path}"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_dirs()
    return settings


settings = get_settings()

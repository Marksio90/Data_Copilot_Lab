"""
Data Copilot Lab - Configuration Management
Centralized configuration using Pydantic Settings
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # Application
    app_name: str = Field(default="Data Copilot Lab", env="APP_NAME")
    app_version: str = Field(default="0.1.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=True, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    # API Configuration
    api_host: str = Field(default="0.0.0.0", env="API_HOST")
    api_port: int = Field(default=8000, env="API_PORT")
    api_reload: bool = Field(default=True, env="API_RELOAD")
    api_workers: int = Field(default=4, env="API_WORKERS")

    # Database
    database_url: str = Field(
        default="sqlite:///./data/data_copilot.db",
        env="DATABASE_URL"
    )
    db_pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")

    # Redis
    redis_host: str = Field(default="localhost", env="REDIS_HOST")
    redis_port: int = Field(default=6379, env="REDIS_PORT")
    redis_db: int = Field(default=0, env="REDIS_DB")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")

    @property
    def redis_url(self) -> str:
        """Construct Redis URL from components"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # Celery
    celery_broker_url: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_BROKER_URL"
    )
    celery_result_backend: str = Field(
        default="redis://localhost:6379/0",
        env="CELERY_RESULT_BACKEND"
    )

    # OpenAI API
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4-turbo-preview", env="OPENAI_MODEL")
    openai_max_tokens: int = Field(default=4000, env="OPENAI_MAX_TOKENS")
    openai_temperature: float = Field(default=0.7, env="OPENAI_TEMPERATURE")

    # File Storage
    storage_type: str = Field(default="local", env="STORAGE_TYPE")
    upload_dir: Path = Field(default=Path("./data/raw"), env="UPLOAD_DIR")
    processed_dir: Path = Field(default=Path("./data/processed"), env="PROCESSED_DIR")
    models_dir: Path = Field(default=Path("./data/models"), env="MODELS_DIR")
    reports_dir: Path = Field(default=Path("./data/reports"), env="REPORTS_DIR")
    max_upload_size: int = Field(default=100, env="MAX_UPLOAD_SIZE")  # MB

    # S3/MinIO
    s3_bucket: Optional[str] = Field(default=None, env="S3_BUCKET")
    s3_region: Optional[str] = Field(default="us-east-1", env="S3_REGION")
    s3_access_key: Optional[str] = Field(default=None, env="S3_ACCESS_KEY")
    s3_secret_key: Optional[str] = Field(default=None, env="S3_SECRET_KEY")
    s3_endpoint: Optional[str] = Field(default=None, env="S3_ENDPOINT")

    # Security & Authentication
    secret_key: str = Field(
        default="your-secret-key-change-this-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(
        default=30,
        env="ACCESS_TOKEN_EXPIRE_MINUTES"
    )
    refresh_token_expire_days: int = Field(
        default=7,
        env="REFRESH_TOKEN_EXPIRE_DAYS"
    )

    # CORS Settings
    cors_origins: List[str] = Field(
        default=["http://localhost:8501", "http://localhost:3000"],
        env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        env="CORS_ALLOW_CREDENTIALS"
    )
    cors_allow_methods: List[str] = Field(default=["*"], env="CORS_ALLOW_METHODS")
    cors_allow_headers: List[str] = Field(default=["*"], env="CORS_ALLOW_HEADERS")

    # ML Configuration
    ml_model_timeout: int = Field(default=300, env="ML_MODEL_TIMEOUT")
    automl_time_budget: int = Field(default=600, env="AUTOML_TIME_BUDGET")
    max_training_samples: int = Field(default=100000, env="MAX_TRAINING_SAMPLES")
    random_state: int = Field(default=42, env="RANDOM_STATE")

    # Data Processing Limits
    max_rows_preview: int = Field(default=1000, env="MAX_ROWS_PREVIEW")
    max_columns: int = Field(default=500, env="MAX_COLUMNS")
    chunk_size: int = Field(default=10000, env="CHUNK_SIZE")

    # Feature Flags
    enable_ai_assistant: bool = Field(default=True, env="ENABLE_AI_ASSISTANT")
    enable_automl: bool = Field(default=True, env="ENABLE_AUTOML")
    enable_deep_learning: bool = Field(default=False, env="ENABLE_DEEP_LEARNING")
    enable_profiling: bool = Field(default=True, env="ENABLE_PROFILING")
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")

    # Monitoring
    sentry_dsn: Optional[str] = Field(default=None, env="SENTRY_DSN")
    enable_metrics: bool = Field(default=False, env="ENABLE_METRICS")

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of {valid_levels}")
        return v.upper()

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment"""
        valid_envs = ["development", "staging", "production", "testing"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of {valid_envs}")
        return v.lower()

    @validator("storage_type")
    def validate_storage_type(cls, v):
        """Validate storage type"""
        valid_types = ["local", "s3", "minio"]
        if v.lower() not in valid_types:
            raise ValueError(f"Storage type must be one of {valid_types}")
        return v.lower()

    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.upload_dir,
            self.processed_dir,
            self.models_dir,
            self.reports_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing"""
        return self.environment == "testing"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance
    Uses lru_cache to avoid reading .env file multiple times
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


# Global settings instance
settings = get_settings()

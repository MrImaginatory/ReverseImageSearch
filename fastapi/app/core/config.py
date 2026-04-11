import os
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Visual Hybrid Search API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths (Defaults based on project structure)
    # Using os.path.abspath to ensure valid paths if not provided in .env
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    
    IMAGES_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Images"))
    MODEL_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "ClipVit"))
    
    # Database
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "root"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5433"
    POSTGRES_DB: str = "clip_search"
    
    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @computed_field
    @property
    def MODEL_ONNX_PATH(self) -> str:
        return os.path.join(self.MODEL_DIR, "model.onnx")
    
    @computed_field
    @property
    def PREPROCESSOR_CONFIG_PATH(self) -> str:
        return os.path.join(self.MODEL_DIR, "preprocessor_config.json")
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()

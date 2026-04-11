import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Visual Hybrid Search API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Paths
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    IMAGES_DIR: str = os.path.join(BASE_DIR, "Images")
    MODEL_DIR: str = os.path.join(BASE_DIR, "ClipVit")
    
    # Model Config
    MODEL_ONNX_PATH: str = os.path.join(MODEL_DIR, "model.onnx")
    PREPROCESSOR_CONFIG_PATH: str = os.path.join(MODEL_DIR, "preprocessor_config.json")
    
    # Database
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "postgres")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "root")
    POSTGRES_SERVER: str = os.getenv("POSTGRES_SERVER", "localhost")
    POSTGRES_PORT: str = os.getenv("POSTGRES_PORT", "5433")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "clip_vector_db")
    
    DATABASE_URL: str = f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_SERVER}:{POSTGRES_PORT}/{POSTGRES_DB}"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    class Config:
        case_sensitive = True

settings = Settings()

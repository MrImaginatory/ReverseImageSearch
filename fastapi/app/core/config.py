from pydantic_settings import BaseSettings
from pydantic import ConfigDict
import os
from typing import List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Hybrid Image Search API"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # DATABASE
    # postgresql+asyncpg://user:pass@host:port/dbname
    DATABASE_URL: str = "postgresql+asyncpg://postgres:root@localhost:5433/clip_search"
    
    # PATHS
    # The project root is 3 levels up from this file (app/core/config.py)
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    IMAGES_DIR: str = os.path.join(BASE_DIR, "Images")
    MODEL_DIR: str = os.path.join(BASE_DIR, "ClipVit")
    
    # SEARCH SETTINGS
    DEFAULT_LIMIT: int = 6
    PATTERN_SCORE_THRESHOLD: float = 0.45
    
    model_config = ConfigDict(case_sensitive=True, env_file=".env")

settings = Settings()

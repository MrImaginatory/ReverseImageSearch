from pydantic import BaseModel
from typing import List, Optional

class SearchMatch(BaseModel):
    filename: str
    total_similarity: float
    pattern_score: float
    color_score: float

class SearchResponse(BaseModel):
    matches: List[SearchMatch]
    total_found: int

class IndexResponse(BaseModel):
    message: str
    indexed_count: int
    total_in_db: int

class StatsResponse(BaseModel):
    total_images: int
    project_name: str
    version: str

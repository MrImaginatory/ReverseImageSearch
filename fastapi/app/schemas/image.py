from pydantic import BaseModel
from typing import List, Optional

class SearchResult(BaseModel):
    filename: str
    total_similarity: float
    semantic_score: float
    color_dist_score: float
    texture_score: float

class IndexStatus(BaseModel):
    status: str
    processed: int
    total: int
    deleted: int

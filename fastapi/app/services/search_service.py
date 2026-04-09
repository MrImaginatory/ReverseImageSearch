from io import BytesIO
import numpy as np
from PIL import Image
from app.repository.image_repo import ImageRepository
from app.services.clip_service import CLIPService
from app.core.config import settings

class SearchService:
    def __init__(self, repo: ImageRepository, clip: CLIPService):
        self.repo = repo
        self.clip = clip

    async def hybrid_search(self, image_bytes: bytes, color_weight: float = 0.5, limit: int = 6):
        # 1. Load image
        img = Image.open(BytesIO(image_bytes))
        
        # 2. Extract features
        query_emb = self.clip.get_embedding(img)
        query_color = self.clip.get_dominant_color(img)
        
        if query_emb is None or query_color is None:
            return []

        # 3. DB Search
        results = await self.repo.search_hybrid(
            query_embedding=query_emb.flatten(),
            query_color=query_color,
            color_weight=color_weight,
            limit=limit,
            threshold=settings.PATTERN_SCORE_THRESHOLD
        )
        
        return results

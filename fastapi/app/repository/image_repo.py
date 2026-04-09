from typing import List, Tuple, Optional
from sqlalchemy import select, func, delete
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.image import ImageEmbedding
import numpy as np

class ImageRepository:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_all_filenames(self) -> List[str]:
        result = await self.db.execute(select(ImageEmbedding.filename))
        return [row[0] for row in result.all()]

    async def get_total_count(self) -> int:
        result = await self.db.execute(select(func.count()).select_from(ImageEmbedding))
        return result.scalar() or 0

    async def save_embedding(self, filename: str, embedding: np.ndarray, color_rgb: Optional[np.ndarray] = None):
        # Check if exists
        result = await self.db.execute(
            select(ImageEmbedding).where(ImageEmbedding.filename == filename)
        )
        db_image = result.scalar_one_or_none()

        if db_image:
            db_image.embedding = embedding.tolist()
            if color_rgb is not None:
                db_image.color_rgb = color_rgb.tolist()
        else:
            db_image = ImageEmbedding(
                filename=filename,
                embedding=embedding.tolist(),
                color_rgb=color_rgb.tolist() if color_rgb is not None else None
            )
            self.db.add(db_image)
        
        await self.db.commit()

    async def delete_by_filename(self, filename: str):
        await self.db.execute(
            delete(ImageEmbedding).where(ImageEmbedding.filename == filename)
        )
        await self.db.commit()

    async def search_hybrid(
        self, 
        query_embedding: np.ndarray, 
        query_color: np.ndarray, 
        color_weight: float = 0.5, 
        limit: int = 6,
        threshold: float = 0.45
    ) -> List[Tuple[str, float, float, float]]:
        """
        Performs hybrid similarity search.
        Returns List of (filename, total_similarity, pattern_score, color_score)
        """
        # Distance = 1 - Similarity
        # Similarity = 1 - Distance
        pattern_dist = ImageEmbedding.embedding.cosine_distance(query_embedding.tolist())
        color_dist = ImageEmbedding.color_rgb.cosine_distance(query_color.tolist())
        
        pattern_score = 1 - pattern_dist
        color_score = 1 - color_dist
        
        # total = pattern_score * ( (1.0 - weight) + (weight * color_score) )
        total_sim = pattern_score * ((1.0 - color_weight) + (color_weight * color_score))
        
        stmt = (
            select(
                ImageEmbedding.filename,
                total_sim.label("total_similarity"),
                pattern_score.label("pattern_score"),
                color_score.label("color_score")
            )
            .where(pattern_score > threshold)
            .order_by(total_sim.desc())
            .limit(limit)
        )
        
        result = await self.db.execute(stmt)
        return [(r.filename, r.total_similarity, r.pattern_score, r.color_score) for r in result.all()]

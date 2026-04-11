import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func, text
from app.models.image import ImageEmbedding, RegionEmbedding, ColorDistribution

class ImageCRUD:
    @staticmethod
    async def get_all_filenames(db: AsyncSession):
        result = await db.execute(select(ImageEmbedding.filename))
        return set(row[0] for row in result.all())

    @staticmethod
    async def get_incomplete_filenames(db: AsyncSession):
        # Find images that don't have region embeddings
        query = text("""
            SELECT e.filename FROM image_embeddings e
            LEFT JOIN region_embeddings r ON e.filename = r.filename
            WHERE r.filename IS NULL
        """)
        result = await db.execute(query)
        return set(row[0] for row in result.all())

    @staticmethod
    async def save_embedding(
        db: AsyncSession, 
        filename: str, 
        embedding: np.ndarray, 
        color_rgb: np.ndarray = None, 
        texture_vec: np.ndarray = None, 
        regions: list = None, 
        color_dist: list = None
    ):
        # 1. Main Embedding
        # In SQLAlchemy 2.0, we use merge or manually check for exists
        stmt = select(ImageEmbedding).where(ImageEmbedding.filename == filename)
        result = await db.execute(stmt)
        db_image = result.scalar_one_or_none()
        
        if db_image:
            db_image.embedding = embedding.flatten()
            db_image.color_rgb = color_rgb.flatten() if color_rgb is not None else None
            db_image.texture_vector = texture_vec.flatten() if texture_vec is not None else None
        else:
            db_image = ImageEmbedding(
                filename=filename,
                embedding=embedding.flatten(),
                color_rgb=color_rgb.flatten() if color_rgb is not None else None,
                texture_vector=texture_vec.flatten() if texture_vec is not None else None
            )
            db.add(db_image)
            
        # 2. Regions
        if regions:
            await db.execute(delete(RegionEmbedding).where(RegionEmbedding.filename == filename))
            for r_type, r_emb in regions:
                db.add(RegionEmbedding(filename=filename, region_type=r_type, embedding=r_emb.flatten()))
                
        # 3. Color Distribution
        if color_dist:
            await db.execute(delete(ColorDistribution).where(ColorDistribution.filename == filename))
            for color, prop in color_dist:
                db.add(ColorDistribution(filename=filename, color=color.flatten(), proportion=prop))
        
        await db.commit()

    @staticmethod
    async def delete_embedding(db: AsyncSession, filename: str):
        await db.execute(delete(ImageEmbedding).where(ImageEmbedding.filename == filename))
        await db.execute(delete(RegionEmbedding).where(RegionEmbedding.filename == filename))
        await db.execute(delete(ColorDistribution).where(ColorDistribution.filename == filename))
        await db.commit()

    @staticmethod
    async def search_hybrid(
        db: AsyncSession,
        query_embedding: np.ndarray,
        query_color: np.ndarray,
        query_texture: np.ndarray = None,
        color_weight: float = 0.3,
        texture_weight: float = 0.2,
        limit: int = 12
    ):
        # Porting the complex CTE from the original database.py
        # Using text() for pgvector operators like <=>
        
        query = text("""
            WITH LocalizedScores AS (
                SELECT filename, MAX(1 - (embedding <=> CAST(:query_emb AS vector))) as best_region_score
                FROM region_embeddings
                GROUP BY filename
            ),
            ColorDistributionScores AS (
                SELECT filename, SUM((1 - (color <=> CAST(:query_color AS vector))) * proportion) as color_dist_score
                FROM color_distribution
                GROUP BY filename
            ),
            BaseMatches AS (
                SELECT e.filename, 
                       (1 - (e.embedding <=> CAST(:query_emb AS vector))) as global_semantic_score,
                       COALESCE(ls.best_region_score, 0) as local_semantic_score,
                       COALESCE(cs.color_dist_score, 0) as color_dist_score,
                       (1 - (e.texture_vector <=> CAST(:query_texture AS vector))) as texture_score
                FROM image_embeddings e
                LEFT JOIN LocalizedScores ls ON e.filename = ls.filename
                LEFT JOIN ColorDistributionScores cs ON e.filename = cs.filename
            )
            SELECT filename, 
                   ( (1.0 - :color_w - :texture_w) * GREATEST(global_semantic_score, local_semantic_score) ) + 
                   (:color_w * color_dist_score) + 
                   (:texture_w * texture_score) AS total_similarity,
                   GREATEST(global_semantic_score, local_semantic_score) as semantic_score,
                   color_dist_score,
                   texture_score
            FROM BaseMatches
            WHERE global_semantic_score > 0.35 OR local_semantic_score > 0.45
            ORDER BY total_similarity DESC
            LIMIT :limit
        """)
        
        result = await db.execute(query, {
            "query_emb": str(query_embedding.flatten().tolist()),
            "query_color": str(query_color.flatten().tolist()),
            "query_texture": str(query_texture.flatten().tolist()) if query_texture is not None else str(query_embedding.flatten().tolist()),
            "color_w": color_weight,
            "texture_w": texture_weight,
            "limit": limit
        })
        
        return result.all()
    
    @staticmethod
    async def get_total_count(db: AsyncSession):
        result = await db.execute(text("SELECT COUNT(*) FROM image_embeddings"))
        return result.scalar()

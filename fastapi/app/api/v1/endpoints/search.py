from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
import io
import numpy as np
from app.db.session import get_db
from app.crud.image_crud import ImageCRUD
from app.services.clip_service import CLIPService, get_clip_service
from app.services.image_service import ImageService
from app.schemas.image import SearchResult
from typing import List

router = APIRouter()

@router.post("/", response_model=List[SearchResult])
async def search_image(
    file: UploadFile = File(...),
    limit: int = Form(6),
    db: AsyncSession = Depends(get_db),
    clip: CLIPService = Depends(get_clip_service)
):
    """
    Performs a Hybrid Search using the uploaded image.
    Automatically tunes weights based on image characteristics (Crop, Detail, etc.).
    """
    # 1. Load and process image
    contents = await file.read()
    query_image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Foreground extraction
    query_fg = ImageService.extract_foreground(query_image)
    
    # 2. Auto-tune weights
    color_boost, texture_boost, strategy = ImageService.auto_tune_weights(query_fg)
    
    # 3. Feature Extraction
    query_emb = clip.get_embedding(query_fg, do_center_crop=False)
    query_colors = ImageService.get_color_distribution(query_fg, k=5)
    query_texture = ImageService.get_texture_vector(query_fg)
    
    if query_emb is None:
        return []

    # 4. Search
    results = await ImageCRUD.search_hybrid(
        db=db,
        query_embedding=query_emb,
        query_color=query_colors[0][0], # Using top color
        query_texture=query_texture,
        color_weight=color_boost,
        texture_weight=texture_boost,
        limit=limit
    )
    
    # 5. Format results
    # Each row: (filename, total_similarity, semantic_score, color_dist_score, texture_score)
    formatted_results = []
    for row in results:
        formatted_results.append({
            "filename": row[0],
            "total_similarity": float(row[1]),
            "semantic_score": float(row[2]),
            "color_dist_score": float(row[3]),
            "texture_score": float(row[4])
        })
        
    return formatted_results

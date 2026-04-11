import os
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from PIL import Image
from app.db.session import get_db
from app.core.config import settings
from app.crud.image_crud import ImageCRUD
from app.services.clip_service import CLIPService, get_clip_service
from app.services.image_service import ImageService
from app.schemas.image import IndexStatus

router = APIRouter()

@router.post("/sync", response_model=IndexStatus)
async def sync_collection(
    db: AsyncSession = Depends(get_db),
    clip: CLIPService = Depends(get_clip_service)
):
    """
    Synchronizes the Images directory with the database.
    This performs feature extraction (AI + Regions + Texture + Color) for new/updated files.
    """
    if not os.path.exists(settings.IMAGES_DIR):
        raise HTTPException(status_code=404, detail=f"Images directory not found at {settings.IMAGES_DIR}")

    all_files = set(f for f in os.listdir(settings.IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    indexed_files = await ImageCRUD.get_all_filenames(db)
    
    new_files = list(all_files - indexed_files)
    migration_files = list((await ImageCRUD.get_incomplete_filenames(db)) & all_files)
    
    files_to_process = list(set(new_files + migration_files))
    deleted_files = list(indexed_files - all_files)
    
    for f in deleted_files:
        await ImageCRUD.delete_embedding(db, f)
        
    for f in files_to_process:
        try:
            path = os.path.join(settings.IMAGES_DIR, f)
            with Image.open(path) as img:
                img = img.convert('RGB')
                
                # 1. AI Embeddings (Global + Regions)
                regions = ImageService.get_image_regions(img)
                region_images = [r[1] for r in regions]
                all_embs = clip.get_embedding(region_images, do_center_crop=True)
                
                global_emb = all_embs[0]
                region_data = [(name, all_embs[idx]) for idx, (name, _) in enumerate(regions) if name != "full"]
                
                # 2. Advanced Color Distribution
                color_dist = ImageService.get_color_distribution(img, k=5)
                top_color = color_dist[0][0]
                
                # 3. Simple Texture (LBP)
                texture_vec = ImageService.get_texture_vector(img)
                
                # Save to DB
                await ImageCRUD.save_embedding(
                    db, f, global_emb, 
                    color_rgb=top_color, 
                    texture_vec=texture_vec,
                    regions=region_data,
                    color_dist=color_dist
                )
        except Exception as e:
            print(f"Error indexing {f}: {e}")
            
    return {
        "status": "completed",
        "processed": len(files_to_process),
        "total": len(all_files),
        "deleted": len(deleted_files)
    }

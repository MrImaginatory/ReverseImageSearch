from typing import AsyncGenerator
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import AsyncSessionLocal
from app.services.clip_service import CLIPService
from app.repository.image_repo import ImageRepository

# Initialize CLIP model once on startup
_clip_service_instance = None

def get_clip_service() -> CLIPService:
    global _clip_service_instance
    if _clip_service_instance is None:
        _clip_service_instance = CLIPService()
    return _clip_service_instance

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

def get_image_repo(db: AsyncSession = Depends(get_db)) -> ImageRepository:
    return ImageRepository(db)

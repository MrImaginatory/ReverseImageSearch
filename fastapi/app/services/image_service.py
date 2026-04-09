import os
import numpy as np
from typing import Callable, Optional
from app.core.config import settings
from app.repository.image_repo import ImageRepository
from app.services.clip_service import CLIPService

class IndexingService:
    def __init__(self, repo: ImageRepository, clip: CLIPService):
        self.repo = repo
        self.clip = clip

    async def synchronize_index(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        """
        Synchronizes the Images directory with the database.
        """
        if not os.path.exists(settings.IMAGES_DIR):
            os.makedirs(settings.IMAGES_DIR)

        # 1. Get current state
        all_files = set(f for f in os.listdir(settings.IMAGES_DIR) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
        indexed_files = set(await self.repo.get_all_filenames())
        
        # 2. Identify changes
        new_files = list(all_files - indexed_files)
        deleted_files = list(indexed_files - all_files)
        
        # 3. Clean up deleted images
        for f in deleted_files:
            await self.repo.delete_by_filename(f)
            
        if not new_files:
            return 0 # Nothing to index

        # 4. Process new images
        total = len(new_files)
        indexed_count = 0
        for i, f in enumerate(new_files):
            path = os.path.join(settings.IMAGES_DIR, f)
            try:
                emb = self.clip.get_embedding(path)
                color_vec = self.clip.get_dominant_color(path)
                
                if emb is not None:
                    await self.repo.save_embedding(f, emb.flatten(), color_rgb=color_vec)
                    indexed_count += 1
            except Exception as e:
                print(f"Error indexing {f}: {e}")
            
            if progress_callback:
                progress_callback(i + 1, total)
                
        return indexed_count

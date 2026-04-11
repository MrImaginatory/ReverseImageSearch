import os
import asyncio
import sys
from PIL import Image
from typing import Set

# Ensure the current directory is in the path so we can import app modules
# This allows running from the Fastapi directory: python batch_index.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.core.config import settings
from app.db.session import engine, SessionLocal
from app.crud.image_crud import ImageCRUD
from app.services.clip_service import CLIPService
from app.services.image_service import ImageService

async def main():
    print("🚀 Starting Batch Indexing Service...")
    
    # Initialize CLIP Service
    print("⏳ Loading CLIP model (this may take a moment)...")
    try:
        clip = CLIPService()
    except Exception as e:
        print(f"❌ Error loading CLIP service: {e}")
        return
    
    if not os.path.exists(settings.IMAGES_DIR):
        print(f"❌ Error: Images directory not found at {settings.IMAGES_DIR}")
        return

    # Scan directory
    extensions = ('.png', '.jpg', '.jpeg', '.webp')
    all_files = sorted([f for f in os.listdir(settings.IMAGES_DIR) if f.lower().endswith(extensions)])
    total_files = len(all_files)
    print(f"📁 Found {total_files} images in {settings.IMAGES_DIR}")

    # Get already indexed files to detect what's missing or incomplete
    async with SessionLocal() as db:
        print("🔍 Checking database for existing records...")
        indexed_on_db = await ImageCRUD.get_all_filenames(db)
        incomplete_on_db = await ImageCRUD.get_incomplete_filenames(db)
    
    # Files that need processing (either new to DB or missing advanced features)
    new_files = [f for f in all_files if f not in indexed_on_db]
    to_reprocess = [f for f in all_files if f in incomplete_on_db]
    
    files_to_process = sorted(list(set(new_files + to_reprocess)))
    count_to_process = len(files_to_process)
    
    print(f"✅ Already indexed: {len(indexed_on_db)}")
    print(f"📦 To process: {count_to_process}")
    
    if count_to_process == 0:
        print("✨ All images are already indexed and complete. Nothing to do!")
        return

    processed_count = 0
    errors_count = 0

    print("\n--- Starting Processing ---")
    for idx, f in enumerate(files_to_process):
        try:
            path = os.path.join(settings.IMAGES_DIR, f)
            print(f"[{idx+1}/{count_to_process}] Processing {f}...", end="\r", flush=True)
            
            # Open and convert image
            with Image.open(path) as img:
                img = img.convert('RGB')
                
                # 1. AI Embeddings (Global + Regions)
                regions = ImageService.get_image_regions(img)
                region_images = [r[1] for r in regions]
                # clip.get_embedding handles the batching internally
                all_embs = clip.get_embedding(region_images, do_center_crop=True)
                
                global_emb = all_embs[0]
                region_data = [(name, all_embs[i]) for i, (name, _) in enumerate(regions) if name != "full"]
                
                # 2. Advanced Color Distribution
                color_dist = ImageService.get_color_distribution(img, k=5)
                top_color = color_dist[0][0]
                
                # 3. Simple Texture (LBP)
                texture_vec = ImageService.get_texture_vector(img)
                
                # Save to DB (new session per image to avoid long-lived transaction issues)
                async with SessionLocal() as db:
                    await ImageCRUD.save_embedding(
                        db, f, global_emb, 
                        color_rgb=top_color, 
                        texture_vec=texture_vec,
                        regions=region_data,
                        color_dist=color_dist
                    )
            
            processed_count += 1
        except Exception as e:
            print(f"\n❌ Error indexing {f}: {e}")
            errors_count += 1
            
    print(f"\n\n🏁 Batch indexing complete!")
    print(f"📊 Successfully Processed: {processed_count}")
    print(f"⚠️ Errors encountered: {errors_count}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n🛑 Indexing interrupted by user. Progress has been saved to the database.")
    except Exception as e:
        print(f"\n\n💥 An unexpected error occurred: {e}")

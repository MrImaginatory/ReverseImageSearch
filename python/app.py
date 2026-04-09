import os
import json
import numpy as np
import sys
from tqdm import tqdm

# Add the streamlit directory to sys.path to import shared logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "streamlit")))

from core import CLIPModel, create_index
from database import DatabaseManager

def main():
    # Paths relative to the script's parent directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    images_dir = os.path.join(base_dir, "Images")
    model_dir = os.path.join(base_dir, "ClipVit")
    
    # Initialize DB
    print("Connecting to PostgreSQL collection...")
    try:
        db = DatabaseManager()
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Make sure your Podman container is running on port 5433.")
        return

    # Load configs
    with open(os.path.join(model_dir, "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
    
    model = CLIPModel(os.path.join(model_dir, "model.onnx"), preprocessor_config)
    
    # Check if index exists in DB
    total_images = db.get_total_count()
    
    if total_images == 0:
        print("Database is empty. Indexing images in 'Images' folder...")
        pbar = tqdm(total=0)
        def update_pbar(current, total):
            if pbar.total == 0:
                pbar.total = total
            pbar.update(1)
        
        create_index(model, images_dir, db, progress_callback=update_pbar)
        pbar.close()
        print(f"Index complete. Total images: {db.get_total_count()}")

    # Search loop
    while True:
        print("\n--- Reverse Image Search (DB Powered) ---")
        query_path = input("Enter image path to search (or 'q' to quit): ").strip().strip('"')
        
        if query_path.lower() == 'q':
            break
            
        if not os.path.exists(query_path):
            print(f"Error: File '{query_path}' not found.")
            continue
            
        print("Analyzing and searching DB...")
        query_emb = model.get_embedding(query_path)
        if query_emb is None:
            continue
            
        # Search via SQL
        results = db.search_similarity(query_emb.flatten(), limit=10)
        
        high_conf = [r for r in results if r[1] >= 0.80]
        rec = [r for r in results if 0.60 <= r[1] < 0.80]
        
        if high_conf:
            print(f"\nFound {len(high_conf)} images with > 80% similarity:")
            for i, (name, score) in enumerate(high_conf[:5]):
                print(f"{i+1}. {name} (Score: {score:.4f})")
        elif rec:
            print("\nNo high-conf matches. Recommendations (60-80%):")
            for i, (name, score) in enumerate(rec[:5]):
                print(f"{i+1}. {name} (Score: {score:.4f})")
        else:
            print("\nNo similar images found.")

if __name__ == "__main__":
    main()

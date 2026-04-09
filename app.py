import os
import pickle
import numpy as np
import json
from tqdm import tqdm
from core import CLIPModel, cosine_similarity

def main():
    images_dir = "Images"
    model_dir = "ClipVit"
    db_file = "embeddings.pkl"
    
    # Load configs
    with open(os.path.join(model_dir, "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
    
    model = CLIPModel(os.path.join(model_dir, "model.onnx"), preprocessor_config)
    
    # Check if index exists or needs update
    embeddings = []
    filenames = []
    
    if os.path.exists(db_file):
        print("Loading existing image index...")
        with open(db_file, "rb") as f:
            data = pickle.load(f)
            embeddings = data["embeddings"]
            filenames = data["filenames"]
    else:
        print("Indexing images in 'Images' folder (this may take a moment)...")
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        for f in tqdm(image_files):
            path = os.path.join(images_dir, f)
            emb = model.get_embedding(path)
            if emb is not None:
                embeddings.append(emb.flatten())
                filenames.append(f)
        
        # Save index
        with open(db_file, "wb") as f:
            pickle.dump({"embeddings": np.array(embeddings), "filenames": filenames}, f)
        embeddings = np.array(embeddings)
        print(f"Indexed {len(filenames)} images.")

    # Search loop
    while True:
        print("\n--- Reverse Image Search ---")
        query_path = input("Enter the path to an image to search (or 'q' to quit): ").strip().strip('"')
        
        if query_path.lower() == 'q':
            break
            
        if not os.path.exists(query_path):
            print(f"Error: File '{query_path}' not found.")
            continue
            
        print("Searching...")
        query_emb = model.get_embedding(query_path)
        if query_emb is None:
            continue
            
        similarities = cosine_similarity(query_emb.flatten(), embeddings)
        sorted_indices = np.argsort(similarities)[::-1]
        
        high_conf_indices = [idx for idx in sorted_indices if similarities[idx] >= 0.80]
        recommended_indices = [idx for idx in sorted_indices if 0.60 <= similarities[idx] < 0.80]
        
        if high_conf_indices:
            print(f"\nFound {len(high_conf_indices)} images with more than 80% similarity:")
            for i, idx in enumerate(high_conf_indices[:5]): 
                print(f"{i+1}. {filenames[idx]} (Score: {similarities[idx]:.4f})")
        elif recommended_indices:
            print("\nNo images found with > 80% similarity. Showing recommendations (60-80% similarity):")
            for i, idx in enumerate(recommended_indices[:5]):
                print(f"{i+1}. {filenames[idx]} (Score: {similarities[idx]:.4f})")
        else:
            print("\nNo similar images found even within the 60% threshold.")

if __name__ == "__main__":
    main()

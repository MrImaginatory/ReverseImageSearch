import os
import pickle
import numpy as np
from PIL import Image
import onnxruntime as ort
from tqdm import tqdm
import sys

class CLIPModel:
    def __init__(self, model_path, preprocessor_config):
        self.session = ort.InferenceSession(model_path)
        self.config = preprocessor_config
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image_path):
        # Load image
        img = Image.open(image_path).convert("RGB")
        
        # Resize - matching shortest edge to 224
        short_edge = self.config.get("size", {}).get("shortest_edge", 224)
        w, h = img.size
        if w < h:
            new_w = short_edge
            new_h = int(h * (short_edge / w))
        else:
            new_h = short_edge
            new_w = int(w * (short_edge / h))
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        
        # Center Crop to 224x224
        crop_size = self.config.get("crop_size", {"height": 224, "width": 224})
        left = (new_w - crop_size["width"]) / 2
        top = (new_h - crop_size["height"]) / 2
        right = (new_w + crop_size["width"]) / 2
        bottom = (new_h + crop_size["height"]) / 2
        img = img.crop((left, top, right, bottom))
        
        # Convert to numpy and normalize
        pixel_values = np.array(img).astype(np.float32)
        
        # Rescale
        rescale_factor = self.config.get("rescale_factor", 1/255.0)
        pixel_values *= rescale_factor
        
        # Normalize
        mean = np.array(self.config.get("image_mean", [0.48145466, 0.4578275, 0.40821073]), dtype=np.float32)
        std = np.array(self.config.get("image_std", [0.26862954, 0.26130258, 0.27577711]), dtype=np.float32)
        pixel_values = (pixel_values - mean) / std
        pixel_values = pixel_values.astype(np.float32)
        
        # Transpose to Channel First (C, H, W)
        pixel_values = pixel_values.transpose(2, 0, 1)
        
        # Add batch dimension
        return np.expand_dims(pixel_values, axis=0)

    def get_embedding(self, image_path):
        try:
            input_tensor = self.preprocess(image_path)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            embedding = outputs[0]
            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None

def cosine_similarity(query_emb, database_embs):
    # Dot product since vectors are normalized
    return np.dot(database_embs, query_emb.T).flatten()

def main():
    images_dir = "Images"
    model_dir = "ClipVit"
    db_file = "embeddings.pkl"
    
    # Load configs
    import json
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
            for i, idx in enumerate(high_conf_indices[:5]): # Show top 5 high-confidence
                print(f"{i+1}. {filenames[idx]} (Score: {similarities[idx]:.4f})")
        elif recommended_indices:
            print("\nNo images found with > 80% similarity. Showing recommendations (60-80% similarity):")
            for i, idx in enumerate(recommended_indices[:5]):
                print(f"{i+1}. {filenames[idx]} (Score: {similarities[idx]:.4f})")
        else:
            print("\nNo similar images found even within the 60% threshold.")

if __name__ == "__main__":
    main()

import os
import pickle
import numpy as np
from PIL import Image
import onnxruntime as ort

class CLIPModel:
    def __init__(self, model_path, preprocessor_config):
        self.session = ort.InferenceSession(model_path)
        self.config = preprocessor_config
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image):
        """
        Preprocesses a PIL Image object.
        """
        img = image.convert("RGB")
        
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

    def get_embedding(self, image):
        """
        Takes a PIL Image or path and returns a normalized embedding.
        """
        if isinstance(image, str):
            image = Image.open(image)
            
        try:
            input_tensor = self.preprocess(image)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            embedding = outputs[0]
            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

def cosine_similarity(query_emb, database_embs):
    # Dot product since vectors are normalized
    return np.dot(database_embs, query_emb.T).flatten()

def create_index(model, images_dir, db_manager, progress_callback=None):
    """
    Scans the images directory, generates embeddings, and saves them to the database.
    """
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    total = len(image_files)
    
    for i, f in enumerate(image_files):
        path = os.path.join(images_dir, f)
        emb = model.get_embedding(path)
        if emb is not None:
            # Save directly to DB
            db_manager.save_embedding(f, emb.flatten())
        
        if progress_callback:
            progress_callback(i + 1, total)
            
    return True

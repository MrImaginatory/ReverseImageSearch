import os
import numpy as np
from PIL import Image
import torch
from transformers import ViTImageProcessor, ViTModel

class ViTModelWrapper:
    def __init__(self, model_path):
        """
        Initializes the Google ViT model using transformers.
        """
        print(f"Loading Google ViT from {model_path}...")
        self.processor = ViTImageProcessor.from_pretrained(model_path)
        self.model = ViTModel.from_pretrained(model_path)
        self.model.eval()
        
        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, image):
        """
        Takes a PIL Image or path and returns a normalized 768-dim embedding.
        Uses the [CLS] token from the last hidden state.
        """
        if isinstance(image, str):
            image = Image.open(image)
            
        try:
            image = image.convert("RGB")
            # Preprocess image
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Google ViT-base output: [batch_size, sequence_length, hidden_size]
                # We take the CLS token (index 0)
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize embedding for cosine similarity
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

def get_dominant_color(image_path):
    """
    Extracts the dominant RGB color from an image.
    Works by resizing the image to a small scale and calculating the median color.
    """
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('RGB')
    else:
        img = image_path.convert('RGB')
    
    # Resize to a small thumbnail to simplify color space
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    
    # Focus on the center
    width, height = img.size
    left = width // 4
    top = height // 4
    right = 3 * width // 4
    bottom = 3 * height // 4
    img_center = img.crop((left, top, right, bottom))
    
    data = np.array(img_center)
    median_rgb = np.median(data, axis=(0, 1))
    return median_rgb / 255.0

def get_texture_vector(image_path):
    """
    Extracts a texture descriptor using Local Binary Patterns (LBP).
    Returns a normalized 32-bin histogram as a vector.
    """
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('L')
    else:
        img = image_path.convert('L')
    
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    data = np.array(img)
    
    data_pad = np.pad(data, 1, mode='edge')
    lbp = np.zeros_like(data, dtype=np.uint16)
    
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = data_pad[1+dy : 1+dy+data.shape[0], 1+dx : 1+dx+data.shape[1]]
        lbp += (neighbor >= data).astype(np.uint16) * (2**i)
    
    hist, _ = np.histogram(lbp, bins=32, range=(0, 255))
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist

def create_index(model, images_dir, db_manager, progress_callback=None):
    """
    Synchronizes the images directory with the database.
    """
    all_files = set(f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    indexed_files = db_manager.get_all_filenames()
    
    new_files = list(all_files - indexed_files)
    deleted_files = list(indexed_files - all_files)
    
    for f in deleted_files:
        db_manager.delete_embedding(f)
        
    if not new_files:
        return True

    total = len(new_files)
    for i, f in enumerate(new_files):
        path = os.path.join(images_dir, f)
        emb = model.get_embedding(path)
        color_vec = get_dominant_color(path)
        texture_vec = get_texture_vector(path)
        
        if emb is not None:
            db_manager.save_embedding(f, emb.flatten(), color_rgb=color_vec, texture_vec=texture_vec)
        
        if progress_callback:
            progress_callback(i + 1, total)
            
    return True

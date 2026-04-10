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
    # 64x64 is small enough to be fast but maintains enough detail
    img = img.resize((64, 64), Image.Resampling.LANCZOS)
    
    # We focus more on the center of the image to avoid background pixels (like rugs/walls)
    width, height = img.size
    left = width // 4
    top = height // 4
    right = 3 * width // 4
    bottom = 3 * height // 4
    img_center = img.crop((left, top, right, bottom))
    
    # Convert to numpy array
    data = np.array(img_center)
    
    # Calculate median color across height and width
    median_rgb = np.median(data, axis=(0, 1))
    
    # Normalize to 0-1 range for the vector
    return median_rgb / 255.0

def get_texture_vector(image_path):
    """
    Extracts a texture descriptor using Local Binary Patterns (LBP).
    Works by comparing relative intensities in a 3x3 neighborhood.
    Returns a normalized 32-bin histogram as a vector.
    """
    if isinstance(image_path, str):
        img = Image.open(image_path).convert('L')
    else:
        img = image_path.convert('L')
    
    # Resize to standard size for pattern consistency
    img = img.resize((256, 256), Image.Resampling.LANCZOS)
    data = np.array(img)
    
    # Vectorized LBP (8 neighbors)
    # Pad to handle edges
    data_pad = np.pad(data, 1, mode='edge')
    lbp = np.zeros_like(data, dtype=np.uint16)
    
    # Neighbor offsets
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for i, (dy, dx) in enumerate(offsets):
        # Slice neighbor area and compare with center
        neighbor = data_pad[1+dy : 1+dy+data.shape[0], 1+dx : 1+dx+data.shape[1]]
        lbp += (neighbor >= data).astype(np.uint16) * (2**i)
    
    # Create histogram and reduce to 32 bins for efficiency
    hist, _ = np.histogram(lbp, bins=32, range=(0, 255))
    
    # Normalize for cosine similarity
    hist = hist.astype(np.float32)
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist

def cosine_similarity(query_emb, database_embs):
    # Dot product since vectors are normalized
    return np.dot(database_embs, query_emb.T).flatten()

def create_index(model, images_dir, db_manager, progress_callback=None):
    """
    Synchronizes the images directory with the database.
    Only processes new images and removes missing ones.
    """
    # 1. Get current state
    all_files = set(f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    indexed_files = db_manager.get_all_filenames()
    
    # 2. Identify changes
    new_files = list(all_files - indexed_files)
    deleted_files = list(indexed_files - all_files)
    
    # 3. Clean up deleted images
    for f in deleted_files:
        db_manager.delete_embedding(f)
        
    if not new_files:
        return True # Nothing to index

    # 4. Process new images
    total = len(new_files)
    for i, f in enumerate(new_files):
        path = os.path.join(images_dir, f)
        emb = model.get_embedding(path)
        color_vec = get_dominant_color(path)
        texture_vec = get_texture_vector(path)
        
        if emb is not None:
            # Save directly to DB with color and texture
            db_manager.save_embedding(f, emb.flatten(), color_rgb=color_vec, texture_vec=texture_vec)
        
        if progress_callback:
            progress_callback(i + 1, total)
            
    return True

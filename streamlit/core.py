import os
import pickle
import numpy as np
from PIL import Image
import onnxruntime as ort
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Clean up KMeans warnings

class CLIPModel:
    def __init__(self, model_path, preprocessor_config):
        self.session = ort.InferenceSession(model_path)
        self.config = preprocessor_config
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image, do_center_crop=True):
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
        
        if do_center_crop:
            # Center Crop to 224x224
            crop_size = self.config.get("crop_size", {"height": 224, "width": 224})
            left = (new_w - crop_size["width"]) / 2
            top = (new_h - crop_size["height"]) / 2
            right = (new_w + crop_size["width"]) / 2
            bottom = (new_h + crop_size["height"]) / 2
            img = img.crop((left, top, right, bottom))
        else:
            # For queries, we want to see the whole image (no crop)
            # Resize exactly to 224x224 to fit the tensor even if aspect ratio changes slightly
            # (Standard for detail/crop search)
            img = img.resize((224, 224), resample=Image.BICUBIC)
        
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
        
        # Add batch dimension if single image
        if pixel_values.ndim == 3:
            return np.expand_dims(pixel_values, axis=0)
        return pixel_values

    def get_embedding(self, image, do_center_crop=True):
        """
        Takes a PIL Image, path, or a LIST of images.
        Returns normalized embeddings.
        """
        if isinstance(image, str):
            image = [Image.open(image)]
        elif not isinstance(image, list):
            image = [image]
            
        try:
            # Process as batch
            tensors = []
            for img in image:
                tensors.append(self.preprocess(img, do_center_crop=do_center_crop))
            
            input_batch = np.concatenate(tensors, axis=0) if len(tensors) > 1 else tensors[0]
            
            outputs = self.session.run([self.output_name], {self.input_name: input_batch})
            embeddings = outputs[0]
            
            # Normalize each in batch
            normalized = []
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                normalized.append(emb / norm if norm > 0 else emb)
            
            return np.array(normalized) if len(normalized) > 1 else normalized[0]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

def get_color_distribution(image, k=5):
    """
    Extracts k dominant colors and their relative proportions using K-Means.
    Returns: List of (rgb_vector, proportion)
    """
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')
    
    # Downsample for performance
    img = img.resize((100, 100), Image.Resampling.LANCZOS)
    data = np.array(img).reshape(-1, 3)
    
    # Perform K-Means
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    labels = kmeans.fit_predict(data)
    centers = kmeans.cluster_centers_
    
    # Calculate proportions
    counts = np.bincount(labels, minlength=k)
    total = len(labels)
    
    distribution = []
    for i in range(len(centers)):
        if counts[i] > 0:
            rgb = centers[i] / 255.0 # Normalize 0-1
            prop = float(counts[i]) / total
            distribution.append((rgb, prop))
            
    # Sort by descending proportion
    distribution.sort(key=lambda x: x[1], reverse=True)
    return distribution

def get_image_regions(image):
    """
    Slices an image into semantic regions:
    - Full image
    - Rule of Thirds (3x3 grid)
    - 4 Quadrants
    - Center Focus
    Returns: List of (region_name, PIL image)
    """
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')
    
    w, h = img.size
    regions = [("full", img)]
    
    # 1. Rule of Thirds (3x3)
    dw, dh = w // 3, h // 3
    for row in range(3):
        for col in range(3):
            box = (col*dw, row*dh, (col+1)*dw, (row+1)*dh)
            regions.append((f"third_{row}_{col}", img.crop(box)))
            
    # 2. Quadrants (2x2)
    qw, qh = w // 2, h // 2
    for row in range(2):
        for col in range(2):
            box = (col*qw, row*qh, (col+1)*qw, (row+1)*qh)
            regions.append((f"quad_{row}_{col}", img.crop(box)))
            
    # 3. Center Focus (Standard Golden Ratio style focal area)
    cw, ch = int(w * 0.618), int(h * 0.618)
    left = (w - cw) // 2
    top = (h - ch) // 2
    regions.append(("center_focus", img.crop((left, top, left+cw, top+ch))))
    
    return regions

def get_dominant_color(image_path):
    """
    Legacy compatibility: Extracts the top color from distribution.
    """
    dist = get_color_distribution(image_path, k=1)
    return dist[0][0] if dist else np.array([0.5, 0.5, 0.5])

def get_texture_vector(image_path):
    """
    Extracts a texture descriptor using Local Binary Patterns (LBP).
    Works on the whole image for consistency across crops.
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
    """
    all_files = set(f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')))
    indexed_files = db_manager.get_all_filenames()
    
    # 2. Identify changes: Missing files AND files needing feature migration
    new_files = list(all_files - indexed_files)
    migration_files = list(db_manager.get_incomplete_filenames() & all_files)
    
    files_to_process = list(set(new_files + migration_files))
    deleted_files = list(indexed_files - all_files)
    
    for f in deleted_files:
        db_manager.delete_embedding(f)
        
    if not files_to_process:
        return True

    total = len(files_to_process)
    for i, f in enumerate(files_to_process):
        try:
            path = os.path.join(images_dir, f)
            img = Image.open(path).convert('RGB')
            
            # 1. AI Embeddings (Global + Regions)
            regions = get_image_regions(img)
            region_images = [r[1] for r in regions]
            
            # Process all regions in one batch call
            all_embs = model.get_embedding(region_images, do_center_crop=True)
            
            global_emb = all_embs[0]
            region_data = []
            for idx, (name, _) in enumerate(regions):
                if name != "full":
                    region_data.append((name, all_embs[idx]))
            
            # 2. Advanced Color Distribution
            color_dist = get_color_distribution(img, k=5)
            top_color = color_dist[0][0] # dominant color for backward compatibility
            
            # 3. Simple Texture (LBP)
            texture_vec = get_texture_vector(img)
            
            # Save all to DB
            db_manager.save_embedding(
                f, 
                global_emb.flatten(), 
                color_rgb=top_color, 
                texture_vec=texture_vec,
                regions=region_data,
                color_dist=color_dist
            )
        except Exception as e:
            print(f"Error indexing {f}: {e}")
            
        if progress_callback:
            progress_callback(i + 1, total)
            
    return True

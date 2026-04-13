import os
import pickle
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel as HFCLIPModel
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore', category=UserWarning) # Clean up KMeans warnings

class CLIPModel:
    def __init__(self, model_path):
        """
        Loads the CLIP model using transformers.
        model_path: Path to the directory containing config.json, pytorch_model.bin, etc.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = HFCLIPModel.from_pretrained(model_path).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_path)

    def preprocess(self, image, do_center_crop=True):
        # The processor handles all preprocessing (resize, crop, normalize)
        # However, to maintain the logic of 'no crop' for queries when requested:
        if not do_center_crop:
             # Just resize to square if we want to avoid the default center crop logic
             image = image.resize((224, 224), resample=Image.BICUBIC)
        
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        return inputs.pixel_values

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
            with torch.no_grad():
                # Process images
                pixel_values = []
                for img in image:
                    pixel_values.append(self.preprocess(img, do_center_crop=do_center_crop))
                
                input_batch = torch.cat(pixel_values, dim=0)
                
                image_features = self.model.get_image_features(pixel_values=input_batch)
                
                # Handle case where output might be an object instead of a tensor
                if not torch.is_tensor(image_features):
                    if hasattr(image_features, "pooler_output"):
                        image_features = image_features.pooler_output
                    else:
                        image_features = image_features[0]

                # Normalize
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                embeddings = image_features.cpu().numpy()
            
            return embeddings if len(embeddings) > 1 else embeddings[0]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

def auto_tune_weights(image):
    """
    Analyzes a query image and returns optimal (color_weight, texture_weight, strategy_name).
    
    Heuristics:
    - Crops/details → low color & texture (semantic-first)
    - Full product shots → moderate color & texture
    - Color-uniform images → higher color weight
    """
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')
    
    w, h = img.size
    aspect = max(w, h) / max(min(w, h), 1)
    
    # 1. Detect if likely a crop (unusual aspect ratio or small area)
    is_likely_crop = aspect > 1.8 or (w * h) < 100_000
    
    # 2. Analyze color concentration
    small = img.resize((64, 64), Image.Resampling.LANCZOS)
    data = np.array(small).reshape(-1, 3).astype(float)
    color_std = np.mean(np.std(data, axis=0))  # Average std across RGB channels
    # Low std = uniform color, High std = varied/patterned
    
    # 3. Edge density (proxy for detail level)
    gray = np.array(img.convert('L').resize((128, 128), Image.Resampling.LANCZOS), dtype=float)
    # Simple Sobel-like edge detection
    dx = np.abs(gray[:, 1:] - gray[:, :-1])
    dy = np.abs(gray[1:, :] - gray[:-1, :])
    edge_density = (np.mean(dx) + np.mean(dy)) / 2.0 / 255.0
    
    # Decision logic
    if is_likely_crop:
        # Crop detected: rely almost entirely on semantic/region matching
        return 0.05, 0.05, "🔍 Crop/Detail Mode"
    elif color_std < 30:
        # Very uniform color (solid fabric): color is a strong signal
        return 0.25, 0.10, "🎨 Color-Dominant Mode"
    elif edge_density > 0.12:
        # Very textured/patterned image: texture is a strong signal
        return 0.10, 0.25, "🧵 Pattern-Heavy Mode"
    else:
        # Balanced full product shot
        return 0.15, 0.10, "⚖️ Balanced Mode"

def extract_foreground(image):
    """
    Detects and removes transparent or white backgrounds.
    Returns a cropped image containing only the foreground subject.
    """
    if isinstance(image, str):
        img = Image.open(image)
    else:
        img = image.copy()
    
    # 1. Check for alpha channel (transparent background - e.g., from rembg)
    if img.mode == 'RGBA':
        alpha = np.array(img)[:, :, 3]
        transparent_ratio = np.mean(alpha < 128)
        if transparent_ratio > 0.05:  # More than 5% transparent
            mask = alpha > 128
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                # Crop to foreground bounding box and convert to RGB
                cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
                # Paste onto white to avoid transparency issues with CLIP
                bg = Image.new('RGB', cropped.size, (255, 255, 255))
                bg.paste(cropped, mask=cropped.split()[3])
                return bg
    
    # 2. Check for white/near-white background
    rgb = np.array(img.convert('RGB'))
    is_bg = np.all(rgb > 240, axis=2)  # Near-white pixels
    bg_ratio = np.mean(is_bg)
    
    if bg_ratio > 0.15:  # More than 15% white background
        mask = ~is_bg
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            return img.convert('RGB').crop((cmin, rmin, cmax + 1, rmax + 1))
    
    return img.convert('RGB')

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
            
            if all_embs is None:
                raise ValueError("Model failed to generate embeddings for this image.")
            
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

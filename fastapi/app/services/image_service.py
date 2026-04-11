import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
import warnings

# Suppress KMeans warnings
warnings.filterwarnings('ignore', category=UserWarning)

class ImageService:
    @staticmethod
    def auto_tune_weights(image: Image.Image):
        img = image.convert('RGB')
        w, h = img.size
        aspect = max(w, h) / max(min(w, h), 1)
        
        is_likely_crop = aspect > 1.8 or (w * h) < 100_000
        
        # Color concentration
        small = img.resize((64, 64), Image.Resampling.LANCZOS)
        data = np.array(small).reshape(-1, 3).astype(float)
        color_std = np.mean(np.std(data, axis=0))
        
        # Edge density
        gray = np.array(img.convert('L').resize((128, 128), Image.Resampling.LANCZOS), dtype=float)
        dx = np.abs(gray[:, 1:] - gray[:, :-1])
        dy = np.abs(gray[1:, :] - gray[:-1, :])
        edge_density = (np.mean(dx) + np.mean(dy)) / 2.0 / 255.0
        
        if is_likely_crop:
            return 0.05, 0.05, "🔍 Crop/Detail Mode"
        elif color_std < 30:
            return 0.25, 0.10, "🎨 Color-Dominant Mode"
        elif edge_density > 0.12:
            return 0.10, 0.25, "🧵 Pattern-Heavy Mode"
        else:
            return 0.15, 0.10, "⚖️ Balanced Mode"

    @staticmethod
    def extract_foreground(image: Image.Image) -> Image.Image:
        img = image.copy()
        if img.mode == 'RGBA':
            alpha = np.array(img)[:, :, 3]
            transparent_ratio = np.mean(alpha < 128)
            if transparent_ratio > 0.05:
                mask = alpha > 128
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if np.any(rows) and np.any(cols):
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    cropped = img.crop((cmin, rmin, cmax + 1, rmax + 1))
                    bg = Image.new('RGB', cropped.size, (255, 255, 255))
                    bg.paste(cropped, mask=cropped.split()[3])
                    return bg
        
        rgb = np.array(img.convert('RGB'))
        is_bg = np.all(rgb > 240, axis=2)
        if np.mean(is_bg) > 0.15:
            mask = ~is_bg
            rows = np.any(mask, axis=1)
            cols = np.any(mask, axis=0)
            if np.any(rows) and np.any(cols):
                rmin, rmax = np.where(rows)[0][[0, -1]]
                cmin, cmax = np.where(cols)[0][[0, -1]]
                return img.convert('RGB').crop((cmin, rmin, cmax + 1, rmax + 1))
        
        return img.convert('RGB')

    @staticmethod
    def get_color_distribution(image: Image.Image, k: int = 5):
        img = image.convert('RGB').resize((100, 100), Image.Resampling.LANCZOS)
        data = np.array(img).reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
        labels = kmeans.fit_predict(data)
        centers = kmeans.cluster_centers_
        
        counts = np.bincount(labels, minlength=k)
        total = len(labels)
        
        distribution = []
        for i in range(len(centers)):
            if counts[i] > 0:
                rgb = centers[i] / 255.0
                prop = float(counts[i]) / total
                distribution.append((rgb, prop))
        
        distribution.sort(key=lambda x: x[1], reverse=True)
        return distribution

    @staticmethod
    def get_texture_vector(image: Image.Image):
        img = image.convert('L').resize((256, 256), Image.Resampling.LANCZOS)
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

    @staticmethod
    def get_image_regions(image: Image.Image):
        img = image.convert('RGB')
        w, h = img.size
        regions = [("full", img)]
        
        # Rule of Thirds
        dw, dh = w // 3, h // 3
        for row in range(3):
            for col in range(3):
                regions.append((f"third_{row}_{col}", img.crop((col*dw, row*dh, (col+1)*dw, (row+1)*dh))))
                
        # Quadrants
        qw, qh = w // 2, h // 2
        for row in range(2):
            for col in range(2):
                regions.append((f"quad_{row}_{col}", img.crop((col*qw, row*qh, (col+1)*qw, (row+1)*qh))))
                
        # Center Focus
        cw, ch = int(w * 0.618), int(h * 0.618)
        left, top = (w - cw) // 2, (h - ch) // 2
        regions.append(("center_focus", img.crop((left, top, left+cw, top+ch))))
        
        return regions

    @staticmethod
    def calibrate_confidence(similarity: float, power: float = 8.0) -> float:
        """
        Calibrates raw cosine similarity into a human-intuitive confidence score.
        Uses a power-law transformation to penalize non-exact matches.
        """
        s = max(0.0, min(1.0, similarity))
        return float(s ** power)

    @staticmethod
    def get_confidence_label(calibrated_score: float) -> str:
        """Categorizes calibrated scores into human-readable labels."""
        if calibrated_score >= 0.98: return "💎 Exact Match"
        if calibrated_score >= 0.85: return "🎯 High Confidence"
        if calibrated_score >= 0.70: return "🔍 Very Similar"
        if calibrated_score >= 0.40: return "🎨 Visual Idea"
        return "🌐 Related Style"

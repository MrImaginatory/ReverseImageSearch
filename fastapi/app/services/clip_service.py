import onnxruntime as ort
import numpy as np
import json
import os
from PIL import Image
from app.core.config import settings

class CLIPService:
    def __init__(self):
        if not os.path.exists(settings.MODEL_ONNX_PATH):
            raise FileNotFoundError(f"Model file not found at {settings.MODEL_ONNX_PATH}")
            
        with open(settings.PREPROCESSOR_CONFIG_PATH, "r") as f:
            self.config = json.load(f)
            
        self.session = ort.InferenceSession(settings.MODEL_ONNX_PATH)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: Image.Image, do_center_crop: bool = True):
        img = image.convert("RGB")
        
        # Resize
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
            crop_size = self.config.get("crop_size", {"height": 224, "width": 224})
            left = (new_w - crop_size["width"]) / 2
            top = (new_h - crop_size["height"]) / 2
            right = (new_w + crop_size["width"]) / 2
            bottom = (new_h + crop_size["height"]) / 2
            img = img.crop((left, top, right, bottom))
        else:
            img = img.resize((224, 224), resample=Image.BICUBIC)
        
        # Convert and Normalize
        pixel_values = np.array(img).astype(np.float32)
        pixel_values *= self.config.get("rescale_factor", 1/255.0)
        
        mean = np.array(self.config.get("image_mean", [0.48145466, 0.4578275, 0.40821073]), dtype=np.float32)
        std = np.array(self.config.get("image_std", [0.26862954, 0.26130258, 0.27577711]), dtype=np.float32)
        pixel_values = (pixel_values - mean) / std
        
        # Channel First
        pixel_values = pixel_values.transpose(2, 0, 1)
        return np.expand_dims(pixel_values, axis=0)

    def get_embedding(self, images, do_center_crop: bool = True):
        if not isinstance(images, list):
            images = [images]
            
        try:
            tensors = [self.preprocess(img, do_center_crop=do_center_crop) for img in images]
            input_batch = np.concatenate(tensors, axis=0)
            
            outputs = self.session.run([self.output_name], {self.input_name: input_batch})
            embeddings = outputs[0]
            
            # Normalize
            normalized = []
            for emb in embeddings:
                norm = np.linalg.norm(emb)
                normalized.append(emb / norm if norm > 0 else emb)
            
            return np.array(normalized) if len(normalized) > 1 else normalized[0]
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

# Global instance
clip_service = None

def get_clip_service():
    global clip_service
    if clip_service is None:
        clip_service = CLIPService()
    return clip_service

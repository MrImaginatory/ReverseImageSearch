import os
import json
import numpy as np
from PIL import Image
import onnxruntime as ort
from app.core.config import settings

class CLIPService:
    def __init__(self):
        model_path = os.path.join(settings.MODEL_DIR, "model.onnx")
        config_path = os.path.join(settings.MODEL_DIR, "preprocessor_config.json")
        
        if not os.path.exists(model_path) or not os.path.exists(config_path):
            raise FileNotFoundError(f"CLIP Model files not found in {settings.MODEL_DIR}")

        self.session = ort.InferenceSession(model_path)
        with open(config_path, "r") as f:
            self.config = json.load(f)
            
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image: Image.Image):
        img = image.convert("RGB")
        
        short_edge = self.config.get("size", {}).get("shortest_edge", 224)
        w, h = img.size
        if w < h:
            new_w = short_edge
            new_h = int(h * (short_edge / w))
        else:
            new_h = short_edge
            new_w = int(w * (short_edge / h))
        img = img.resize((new_w, new_h), resample=Image.BICUBIC)
        
        crop_size = self.config.get("crop_size", {"height": 224, "width": 224})
        left = (new_w - crop_size["width"]) / 2
        top = (new_h - crop_size["height"]) / 2
        right = (new_w + crop_size["width"]) / 2
        bottom = (new_h + crop_size["height"]) / 2
        img = img.crop((left, top, right, bottom))
        
        pixel_values = np.array(img).astype(np.float32)
        rescale_factor = self.config.get("rescale_factor", 1/255.0)
        pixel_values *= rescale_factor
        
        mean = np.array(self.config.get("image_mean", [0.48145466, 0.4578275, 0.40821073]), dtype=np.float32)
        std = np.array(self.config.get("image_std", [0.26862954, 0.26130258, 0.27577711]), dtype=np.float32)
        pixel_values = (pixel_values - mean) / std
        
        pixel_values = pixel_values.transpose(2, 0, 1)
        return np.expand_dims(pixel_values, axis=0)

    def get_embedding(self, image_data):
        if isinstance(image_data, str):
            image = Image.open(image_data)
        elif not isinstance(image_data, Image.Image):
            image = Image.open(image_data)
        else:
            image = image_data
            
        try:
            input_tensor = self.preprocess(image)
            outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
            embedding = outputs[0]
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def get_dominant_color(self, image_data):
        if isinstance(image_data, str):
            img = Image.open(image_data).convert('RGB')
        elif not isinstance(image_data, Image.Image):
            img = Image.open(image_data).convert('RGB')
        else:
            img = image_data.convert('RGB')
        
        img = img.resize((64, 64), Image.Resampling.LANCZOS)
        width, height = img.size
        left, top, right, bottom = width // 4, height // 4, 3 * width // 4, 3 * height // 4
        img_center = img.crop((left, top, right, bottom))
        
        data = np.array(img_center)
        median_rgb = np.median(data, axis=(0, 1))
        return median_rgb / 255.0

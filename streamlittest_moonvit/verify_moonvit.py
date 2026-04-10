import os
import sys
# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from core import MoonViTModelWrapper
from PIL import Image
import numpy as np
import torch

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "moonshotaiMoonViT-SO-400M")
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "..", "Images", "imgi_100_ynf-faux-georgette-kesh492-1458-kurti-wholesale-designer-kurtis-embroidered-kurtis-kurti-with-palazzo-manufacturer-2026-04-09_15_45_18.jpg")

try:
    print("Initializing MoonViT model...")
    # Some models might need float32 for inference on CPU
    model = MoonViTModelWrapper(MODEL_DIR)
    print("Model loaded.")
    
    print(f"Opening image: {IMAGE_PATH}")
    img = Image.open(IMAGE_PATH)
    
    print("Extracting embedding...")
    emb = model.get_embedding(img)
    
    if emb is not None:
        print(f"Success! Embedding shape: {emb.shape}")
        print(f"First 10 values: {emb.flatten()[:10]}")
        print(f"L2 Norm: {np.linalg.norm(emb)}")
        
        # Check if dimension is 1152
        if emb.shape[-1] == 1152:
            print("Dimension verified: 1152")
        else:
            print(f"Dimension mismatch: expected 1152, got {emb.shape[-1]}")
    else:
        print("Failed to extract embedding.")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

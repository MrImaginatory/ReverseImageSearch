import sys
import os
import numpy as np
from PIL import Image

# Add streamlit dir to path
sys.path.append(os.path.join(os.getcwd(), 'streamlit'))

from core import get_texture_vector

def test_texture_extraction():
    # Create a dummy image with a pattern
    data = np.zeros((100, 100), dtype=np.uint8)
    data[::2, ::2] = 255 # Checkerboard pattern
    img = Image.fromarray(data)
    
    vec = get_texture_vector(img)
    print(f"Texture vector shape: {vec.shape}")
    print(f"Texture vector (first 5): {vec[:5]}")
    
    assert vec.shape == (32,), f"Expected shape (32,), got {vec.shape}"
    assert np.isclose(np.linalg.norm(vec), 1.0), "Vector should be normalized"
    print("Verification successful!")

if __name__ == "__main__":
    test_texture_extraction()

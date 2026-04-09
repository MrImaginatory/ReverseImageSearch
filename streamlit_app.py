import streamlit as st
import os
import pickle
import numpy as np
import json
from PIL import Image
from core import CLIPModel, cosine_similarity

# Set page config
st.set_page_config(page_title="Visual Reverse Image Search", layout="wide")

st.title("🛍️ Reverse Image Search")
st.markdown("Upload an image to find similar items in the collection.")

# Directories
IMAGES_DIR = "Images"
MODEL_DIR = "ClipVit"
DB_FILE = "embeddings.pkl"

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
    return CLIPModel(os.path.join(MODEL_DIR, "model.onnx"), preprocessor_config)

@st.cache_data
def load_index():
    if not os.path.exists(DB_FILE):
        return None, None
    with open(DB_FILE, "rb") as f:
        data = pickle.load(f)
        return data["embeddings"], data["filenames"]

# Load model and index
model = load_model()
embeddings, filenames = load_index()

if embeddings is None:
    st.error("Index not found. Please run `app.py` first to generate the image database.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a query image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    # Display query image
    query_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Query Image")
        st.image(query_image, use_container_width=True)
    
    with col2:
        st.subheader("Search Results")
        with st.spinner("Analyzing image and searching collection..."):
            query_emb = model.get_embedding(query_image)
            
            if query_emb is not None:
                similarities = cosine_similarity(query_emb.flatten(), embeddings)
                sorted_indices = np.argsort(similarities)[::-1]
                
                high_conf_indices = [idx for idx in sorted_indices if similarities[idx] >= 0.80]
                recommended_indices = [idx for idx in sorted_indices if 0.60 <= similarities[idx] < 0.80]
                
                if high_conf_indices:
                    st.success(f"Found {len(high_conf_indices)} images with > 80% similarity")
                    
                    # Display in grid
                    cols = st.columns(3)
                    for i, idx in enumerate(high_conf_indices[:6]): # Show top 6
                        with cols[i % 3]:
                            img_path = os.path.join(IMAGES_DIR, filenames[idx])
                            st.image(img_path, caption=f"{filenames[idx]} (Score: {similarities[idx]:.2f})", use_container_width=True)
                
                elif recommended_indices:
                    st.info("No high-confidence matches found (>80%). Showing recommendations (60-80% similarity):")
                    
                    cols = st.columns(3)
                    for i, idx in enumerate(recommended_indices[:6]):
                        with cols[i % 3]:
                            img_path = os.path.join(IMAGES_DIR, filenames[idx])
                            st.image(img_path, caption=f"{filenames[idx]} (Score: {similarities[idx]:.2f})", use_container_width=True)
                
                else:
                    st.warning("No similar images found even within the 60% threshold.")
            else:
                st.error("Could not process the uploaded image.")

st.sidebar.info(f"Total images in collection: {len(filenames)}")
st.sidebar.markdown("---")
st.sidebar.caption("Powered by CLIP ViT-B-32 ONNX")

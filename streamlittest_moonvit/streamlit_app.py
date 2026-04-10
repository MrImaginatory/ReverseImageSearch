import streamlit as st
import os
from PIL import Image
import numpy as np
from core import MoonViTModelWrapper, create_index, get_dominant_color, get_texture_vector
from database import DatabaseManager

# Set page config
st.set_page_config(page_title="Visual Hybrid Image Search (Moonshot MoonViT)", layout="wide")

st.title("🛍️ Hybrid Reverse Image Search")
st.markdown("Powered by **Moonshot MoonViT-SO-400M** (1152-dim) + **Color & Texture Analysis**.")

# Paths
# Default paths for local running
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Allow override via environment variables (useful for Docker)
IMAGES_DIR = os.getenv("IMAGES_DIR", os.path.join(BASE_DIR, "Images"))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "models", "moonshotaiMoonViT-SO-400M"))

@st.cache_resource
def load_model():
    return MoonViTModelWrapper(MODEL_DIR)

@st.cache_resource
def get_db():
    return DatabaseManager()

# Init components
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Check if model path exists and dependencies are installed.")
    st.stop()

try:
    db = get_db()
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.info("Ensure your PostgreSQL container is running on port 5433.")
    st.stop()

# Sidebar
st.sidebar.title("Search Controls")
total_count = db.get_total_count()
st.sidebar.info(f"Connected to PostgreSQL. Total images: {total_count}")

# Priority Sliders
st.sidebar.markdown("### Search Priority")
color_boost = st.sidebar.slider(
    "Color Boost",
    min_value=0.0,
    max_value=0.5,
    value=0.2,
    step=0.05,
    help="How much color affects the final ranking."
)

texture_boost = st.sidebar.slider(
    "Texture/Pattern Refinement",
    min_value=0.0,
    max_value=0.5,
    value=0.15,
    step=0.05,
    help="How much surface texture affects the ranking."
)

st.sidebar.markdown("---")
st.sidebar.title("Collection Management")
if st.sidebar.button("🔄 Update DB Index"):
    st.sidebar.warning("Processing collection with Moonshot MoonViT (1152-dim)...")
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    
    def update_progress(current, total):
        progress = current / total
        progress_bar.progress(progress)
        status_text.text(f"Processing: {current}/{total}")
        
    create_index(model, IMAGES_DIR, db, progress_callback=update_progress)
    
    st.sidebar.success("Database updated successfully!")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.caption("Moonshot MoonViT-SO-400M | 1152 Dimensions")

if total_count == 0:
    st.warning("The database is currently empty. Please upload images to 'Images/' and click 'Update DB Index'.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Choose a query image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    query_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Query Image")
        st.image(query_image, width=400)
        
        query_color = get_dominant_color(query_image)
        st.markdown("**Extracted Dominant Color:**")
        hex_color = '#%02x%02x%02x' % tuple((query_color * 255).astype(int))
        st.markdown(f'<div style="background-color:{hex_color}; width:100%; height:40px; border-radius:5px; border:1px solid #ddd;"></div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("Results")
        with st.spinner("Executing Moonshot MoonViT hybrid search..."):
            query_emb = model.get_embedding(query_image)
            
            if query_emb is not None:
                query_texture = get_texture_vector(query_image)
                results = db.search_hybrid(
                    query_emb.flatten(), 
                    query_color, 
                    query_texture=query_texture,
                    color_weight=color_boost, 
                    texture_weight=texture_boost,
                    limit=6
                )
                
                if results:
                    st.success(f"Matched {len(results)} items using Moonshot-SO Engine")
                    
                    # Display top result
                    top_result = results[0]
                    other_results = results[1:]
                    
                    name, total_score, p_score, c_score, t_score = top_result
                    img_path = os.path.join(IMAGES_DIR, name)
                    
                    st.markdown("### 🏆 Founded Product")
                    f_col1, f_col2 = st.columns([1, 1])
                    with f_col1:
                        st.image(img_path, width=400)
                    with f_col2:
                        st.info(f"**Filename:** {name}")
                        st.metric("Overall Match", f"{total_score:.2%}")
                        st.progress(total_score)
                        st.caption(f"MoonViT: {p_score:.2f} | Color: {c_score:.2f} | Texture: {t_score:.2f}")

                    st.markdown("---")
                    
                    # Display similar results
                    if other_results:
                        st.markdown("### 🔍 Similar Products")
                        cols = st.columns(3)
                        for i, (name, total_score, p_score, c_score, t_score) in enumerate(other_results[:5]):
                            with cols[i % 3]:
                                img_path = os.path.join(IMAGES_DIR, name)
                                st.image(img_path, width=280)
                                st.markdown(f"**Match: {total_score:.1%}**")
                                st.caption(f"P: {p_score:.2f} | C: {c_score:.2f} | T: {t_score:.2f}")
                else:
                    st.warning("No matches found.")
            else:
                st.error("Processing failed.")

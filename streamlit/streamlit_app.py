import streamlit as st
import os
import json
from PIL import Image
import numpy as np
from core import CLIPModel, create_index, get_color_distribution, get_texture_vector
from database import DatabaseManager

# Set page config
st.set_page_config(page_title="Visual Hybrid Image Search", layout="wide")

st.title("🛍️ Hybrid Reverse Image Search")
st.markdown("Combines **Pattern Recognition** (AI) and **Color Analysis** for more accurate results.")

# Paths relative to the script's parent directory
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(PARENT_DIR, "Images")
MODEL_DIR = os.path.join(PARENT_DIR, "ClipVit")

@st.cache_resource
def load_model():
    with open(os.path.join(MODEL_DIR, "preprocessor_config.json"), "r") as f:
        preprocessor_config = json.load(f)
    return CLIPModel(os.path.join(MODEL_DIR, "model.onnx"), preprocessor_config)

@st.cache_resource
def get_db():
    return DatabaseManager()

# Init components
model = load_model()
try:
    db = get_db()
except Exception as e:
    st.error(f"Error connecting to database: {e}")
    st.info("Ensure your Podman container is running on port 5433.")
    st.code("podman-compose up -d")
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
    # default to 0.1 for a nice default balance
    value=0.15,
    step=0.05,
    help="How much surface texture (stripes, dots, etc.) affects the ranking."
)

total_boost = color_boost + texture_boost
if total_boost > 0:
    st.sidebar.caption(f"⚖️ **Hybrid Mode**: Refined by {color_boost:.0%} Color & {texture_boost:.0%} Texture.")
else:
    st.sidebar.caption("🎯 Focusing strictly on **Semantic Match**.")

st.sidebar.markdown("---")
st.sidebar.title("Collection Management")
if st.sidebar.button("🔄 Update DB Index"):
    st.sidebar.warning("Processing collection (AI + Color)...")
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
st.sidebar.caption("Powered by CLIP + pgvector Hybrid Engine")

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
        st.image(query_image, width=400) # Use explicit width or 'stretch'
        
        # Show extraction preview (Proportional Palette)
        query_colors = get_color_distribution(query_image, k=5)
        st.markdown("**Core Color Palette (Top 5):**")
        
        palette_html = '<div style="display: flex; width: 100%; height: 40px; border-radius: 5px; overflow: hidden; border: 1px solid #ddd;">'
        for color, prop in query_colors:
            hex_c = '#%02x%02x%02x' % tuple((color * 255).astype(int))
            palette_html += f'<div style="background-color:{hex_c}; width:{prop*100}%; height:100%;" title="{prop:.1%}"></div>'
        palette_html += '</div>'
        st.markdown(palette_html, unsafe_allow_html=True)
    
    with col2:
        st.subheader("Results")
        with st.spinner("Executing hybrid search..."):
            query_emb = model.get_embedding(query_image, do_center_crop=False)
            
            if query_emb is not None:
                # Advanced Localized Hybrid Search
                query_texture = get_texture_vector(query_image)
                results = db.search_hybrid(
                    query_emb.flatten() if query_emb.ndim > 1 else query_emb, 
                    query_colors, 
                    query_texture=query_texture,
                    color_weight=color_boost, 
                    texture_weight=texture_boost,
                    limit=6
                )
                
                if results:
                    # Separate the top result
                    top_result = results[0]
                    other_results = results[1:]
                    
                    st.success(f"Matched {len(results)} items using Hybrid Ranking")
                    
                    # 1. Display Founded Product (85% confidence threshold)
                    name, total_score, p_score, c_score, t_score = top_result
                    img_path = os.path.join(IMAGES_DIR, name)
                    
                    st.markdown("### 🏆 Founded Product")
                    if total_score >= 0.75:
                        f_col1, f_col2 = st.columns([1, 1])
                        with f_col1:
                            st.image(img_path, width=400)
                        with f_col2:
                            st.info(f"**Filename:** {name}")
                            st.metric("Hybrid Match", f"{total_score:.2%}")
                            st.metric("AI Semantic Match", f"{p_score:.2%}")
                            st.progress(total_score)
                            st.caption(f"Pattern Score: {p_score:.2f} | Color: {c_score:.2f} | Texture: {t_score:.2f}")
                    else:
                        st.warning("⚠️ No exact product match found (Score below 75% confidence)")
                        st.info("Try reducing 'Color Boost' or 'Texture Refinement' sliders if you are using a cropped/detailed image.")

                    st.markdown("---")
                    
                    # 2. Display Similar Products (Max 5)
                    if other_results:
                        st.markdown("### 🔍 Similar Products")
                        other_results = other_results[:5] # Enforce max 5
                        cols = st.columns(3)
                        for i, (name, total_score, p_score, c_score, t_score) in enumerate(other_results):
                            with cols[i % 3]:
                                img_path = os.path.join(IMAGES_DIR, name)
                                st.image(img_path, width=280)
                                st.markdown(f"**Match: {total_score:.1%}**")
                                st.caption(f"P: {p_score:.2f} | C: {c_score:.2f} | T: {t_score:.2f}")
                else:
                    st.warning("No matches found.")
            else:
                st.error("Processing failed.")

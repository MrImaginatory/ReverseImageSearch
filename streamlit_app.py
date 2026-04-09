import streamlit as st
import os
import json
from PIL import Image
from core import CLIPModel, create_index
from database import DatabaseManager

# Set page config
st.set_page_config(page_title="Visual Reverse Image Search", layout="wide")

st.title("🛍️ Reverse Image Search")
st.markdown("Upload an image to find similar items in the collection (PostgreSQL Powered).")

# Directories
IMAGES_DIR = "Images"
MODEL_DIR = "ClipVit"

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
st.sidebar.title("Collection Management")
total_count = db.get_total_count()
st.sidebar.info(f"Connected to PostgreSQL. Total images: {total_count}")

if st.sidebar.button("🔄 Update DB Index"):
    st.sidebar.warning("Indexing collection... Please wait.")
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
st.sidebar.caption("Powered by CLIP + pgvector")

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
        st.image(query_image, use_container_width=True)
    
    with col2:
        st.subheader("Search Results")
        with st.spinner("Searching PostgreSQL database..."):
            query_emb = model.get_embedding(query_image)
            
            if query_emb is not None:
                # Search via SQL
                results = db.search_similarity(query_emb.flatten(), limit=12)
                
                high_conf = [r for r in results if r[1] >= 0.80]
                rec = [r for r in results if 0.60 <= r[1] < 0.80]
                
                if high_conf:
                    st.success(f"Found {len(high_conf)} high-confidence matches (>80%)")
                    cols = st.columns(3)
                    for i, (name, score) in enumerate(high_conf[:6]):
                        with cols[i % 3]:
                            img_path = os.path.join(IMAGES_DIR, name)
                            st.image(img_path, caption=f"{name} ({score:.2f})", use_container_width=True)
                
                elif rec:
                    st.info("Showing recommendations (60-80% similarity):")
                    cols = st.columns(3)
                    for i, (name, score) in enumerate(rec[:6]):
                        with cols[i % 3]:
                            img_path = os.path.join(IMAGES_DIR, name)
                            st.image(img_path, caption=f"{name} ({score:.2f})", use_container_width=True)
                
                else:
                    st.warning("No similar images found.")
            else:
                st.error("Processing failed.")

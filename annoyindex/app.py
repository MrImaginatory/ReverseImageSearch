# Read all images from the folder
# Create embeddings for each image
# Build Annoy index using embeddings
# Read Test image
# Create embeddings for test image
# Search Annoy index for nearest neighbours



import gradio as gr
from PIL import Image
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from sklearn.neighbors import NearestNeighbors
import torch
import os

# Initialize the CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Configure directories
UPLOAD_FOLDER = 'static/uploads'
IMAGE_FOLDER = 'static/images'

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)

# Function to extract image embedding using CLIP
def extract_image_embedding(image_path):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)
    return embeddings.squeeze().numpy()



# Pre-build the neighbor model
image_paths = []
all_embeddings = []

print(f"Indexing images from {IMAGE_FOLDER}...")

files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
if not files:
    print("Warning: No images found in static/images. Please add some images.")
else:
    for i, filename in enumerate(files):
        image_path = os.path.join(IMAGE_FOLDER, filename)
        try:
            print(f"[{i+1}/{len(files)}] Indexing {filename}...", flush=True)
            embedding = extract_image_embedding(image_path)
            all_embeddings.append(embedding)
            image_paths.append(image_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}", flush=True)

if all_embeddings:
    # Use NearestNeighbors with cosine metric
    nn_model = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute')
    nn_model.fit(np.array(all_embeddings))
    print(f"Successfully indexed {len(all_embeddings)} images.")
else:
    nn_model = None
    print("No images indexed.")

# Function to find similar images
def find_similar_images(image_path, num_matches=5):
    if nn_model is None:
        return []
    
    embedding = extract_image_embedding(image_path)
    
    # kneighbors expects a 2D array
    distances, indices = nn_model.kneighbors([embedding], n_neighbors=min(num_matches, len(image_paths)))
    
    # distances and indices are 2D arrays (1, num_matches)
    similar_images = []
    for i in range(len(indices[0])):
        idx = indices[0][i]
        dist = distances[0][i]
        similar_images.append({"path": image_paths[idx], "distance": dist})
        
    return similar_images



# Function to display similar images in Gradio
def search_similar_images(uploaded_image):
    # Save the uploaded image
    uploaded_image_path = os.path.join(UPLOAD_FOLDER, "uploaded_image.jpg")
    uploaded_image.save(uploaded_image_path, quality=100)  # Save with maximum quality

    # Find similar images
    similar_images = find_similar_images(uploaded_image_path)

    # Prepare the list of image paths and distances for Gradio
    results = []
    for sim_img in similar_images:
        try:
            image = Image.open(sim_img['path'])
            results.append((image, f"Distance: {sim_img['distance']:.4f}"))
        except Exception as e:
            print(f"Error loading result image {sim_img['path']}: {e}")
    
    return results

# Create the Gradio interface
iface = gr.Interface(
    fn=search_similar_images,
    inputs=gr.Image(type="pil", label="Upload an Image"),
    outputs=gr.Gallery(label="Similar Images"),
    title="Similar Image Search Engine (sklearn)",
    description="Upload an image to find similar images from the dataset."
)

# Launch the Gradio app
if __name__ == "__main__":
    if not image_paths:
        print("Warning: The app is starting with an empty index. Search will not work until images are added to static/images.")
    iface.launch(debug=True)
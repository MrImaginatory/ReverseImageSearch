# Streamlit Web UI Module

Backend modules for the Hybrid Reverse Image Search web interface.

---

## Module Structure

```
streamlit/
├── __init__.py              # Package marker (empty)
├── core.py                  # CLIP model & feature extraction
├── database.py              # PostgreSQL/pgvector operations
└── streamlit_app.py          # Web interface application
```

---

## Files Overview

### core.py

Core AI functionality for image feature extraction:

| Class/Function | Description |
|----------------|-------------|
| `CLIPModel` | ONNX-based CLIP ViT-B/32 encoder |
| `preprocess()` | Image preprocessing (resize, center crop, normalize) |
| `get_embedding()` | Generate 512-dimensional embeddings |
| `auto_tune_weights()` | Detect query type and return optimal search weights |
| `extract_foreground()` | Remove transparent/white backgrounds |
| `get_color_distribution()` | K-Means color palette extraction (k=5) |
| `get_texture_vector()` | LBP texture histogram (32-dim) |
| `get_image_regions()` | Generate region embeddings (3x3, quadrants, center) |
| `create_index()` | Batch index images to database |
| `cosine_similarity()` | Vector similarity computation |

### database.py

PostgreSQL + pgvector database manager:

| Method | Description |
|-------|-------------|
| `DatabaseManager.__init__()` | Connect to PostgreSQL |
| `_init_db()` | Initialize schema and extensions |
| `save_embedding()` | Store image features |
| `delete_embedding()` | Remove image features |
| `get_all_filenames()` | List indexed images |
| `search_hybrid()` | Multi-modal similarity search |
| `get_incomplete_filenames()` | Find images needing re-indexing |
| `get_total_count()` | Total indexed images |

### streamlit_app.py

Streamlit web interface with:

- Image upload and query
- Auto-tuned search strategy detection
- Color palette visualization
- Hybrid search results (semantic + color + texture)
- Database index management

---

## Usage

### Run Web Interface

```bash
cd streamlit
streamlit run streamlit_app.py
```

### Import in Python

```python
from streamlit.core import CLIPModel, create_index
from streamlit.database import DatabaseManager

# Initialize model
model = CLIPModel("path/to/model.onnx", config)

# Initialize database
db = DatabaseManager()

# Index images
create_index(model, "Images/", db)
```

---

## Dependencies

See `requirements.txt` for the complete list.

---

## Database Requirements

- PostgreSQL 16+ with pgvector extension
- Running on `localhost:5433`
- Database: `clip_vector_db`
- User: `postgres` / Password: `root`

---

## Integration with Parent Project

This module is imported by:

1. **python/app.py** - CLI interface
2. **streamlit_app.py** - Web interface

Both reference it via `sys.path` manipulation or relative imports.

---

## API Reference

### CLIPModel

```python
class CLIPModel:
    def __init__(self, model_path: str, preprocessor_config: dict):
        """Load ONNX model."""
    
    def get_embedding(self, image, do_center_crop=True) -> np.ndarray:
        """Generate 512-dim embedding. Accepts PIL Image, path, or list."""
    
    def preprocess(self, image, do_center_crop=True) -> np.ndarray:
        """Preprocess for CLIP model."""
```

### DatabaseManager

```python
class DatabaseManager:
    def __init__(self, host="localhost", port=5433, user="postgres", 
                 password="root", dbname="clip_vector_db"):
        """Connect to PostgreSQL."""
    
    def save_embedding(self, filename: str, embedding, color_rgb=None, 
                     texture_vec=None, regions=None, color_dist=None):
        """Save all features for an image."""
    
    def search_hybrid(self, query_embedding, query_color_dist, 
                     query_texture=None, color_weight=0.3, 
                     texture_weight=0.2, limit=12) -> list:
        """Execute hybrid search."""
```

### Feature Extraction Functions

```python
def auto_tune_weights(image) -> tuple:
    """Returns (color_weight, texture_weight, strategy_name)."""

def extract_foreground(image) -> Image.Image:
    """Strip transparent/white backgrounds."""

def get_color_distribution(image, k=5) -> list:
    """Returns [(rgb_vector, proportion), ...]."""

def get_texture_vector(image) -> np.ndarray:
    """Returns 32-dim LBP histogram."""

def get_image_regions(image) -> list:
    """Returns [(region_name, PIL.Image), ...]."""

def create_index(model, images_dir, db_manager, progress_callback=None):
    """Batch index directory to database."""
```

---

## Configuration

### Model Configuration

Located in `../ClipVit/preprocessor_config.json`:

```json
{
  "size": {"shortest_edge": 224},
  "crop_size": {"height": 224, "width": 224},
  "rescale_factor": 0.00392156862745098,
  "image_mean": [0.48145466, 0.4578275, 0.40821073],
  "image_std": [0.26862954, 0.26130258, 0.27577711]
}
```

### Database Schema

Expected tables (auto-created):

```sql
CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    embedding VECTOR(512) NOT NULL,
    color_rgb VECTOR(3),
    texture_vector VECTOR(32)
);

CREATE TABLE region_embeddings (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    region_type TEXT NOT NULL,
    embedding VECTOR(512) NOT NULL
);

CREATE TABLE color_distribution (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    color VECTOR(3) NOT NULL,
    proportion FLOAT NOT NULL
);
```

---

## Performance

| Operation | Time |
|-----------|------|
| Single embedding generation | ~50ms |
| Full image indexing (all features) | ~1-2s |
| Hybrid search (top 10) | <100ms |

---

## Error Handling

All exceptions are caught and logged with appropriate messages. Check:

1. Database connection - ensure PostgreSQL running
2. Model file exists - check `../ClipVit/model.onnx`
3. Images directory - check `../Images/` exists and has valid images
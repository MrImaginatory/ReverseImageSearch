# Hybrid Reverse Image Search System

A production-grade image similarity search engine combining **CLIP semantic embeddings**, **color distribution analysis**, and **texture matching** for accurate visual product retrieval.

---

## Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Query Image   │────▶│  Python CLI     │────▶│  Streamlit UI   │
│   (User Upload) │     │  (app.py)       │     │  (streamlit_app)│
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                │                      │
                                ▼                      ▼
                    ┌───────────────────┐    ┌───────────────────┐
                    │   CLIP ONNX Model │    │   CLIP ONNX Model │
                    │   (512-dim emb)   │    │   (512-dim emb)   │
                    └─────────┬─────────┘    └─────────┬─────────┘
                              │                      │
                              ▼                      ▼
                    ┌───────────────────────────────────────────────┐
                    │              Feature Extraction               │
                    │  ┌─────────────┬─────────────┬─────────────┐  │
                    │  │  Global     │   Region    │   Color     │  │
                    │  │  Embedding  │  Embeddings │  Palette    │  │
                    │  └─────────────┴─────────────┴─────────────┘  │
                    │  ┌─────────────────────────────────────────┐  │
                    │  │  Texture Vector (LBP 32-dim histogram)  │  │
                    │  └─────────────────────────────────────────┘  │
                    └──────────────────────┬──────────────────────┘
                                           │
                    ┌───────────────────────▼───────────────────────┐
                    │           PostgreSQL + pgvector              │
                    │  ┌─────────────┬─────────────┬─────────────┐  │
                    │  │ image_      │ region_     │ color_      │  │
                    │  │ embeddings  │ embeddings  │ distribution│  │
                    │  │ (VECTOR 512)│ (VECTOR 512)│ (VECTOR 3)  │  │
                    │  └─────────────┴─────────────┴─────────────┘  │
                    └───────────────────────────────────────────────┘
```

---

## Directory Structure

```
Image/
├── python/                  # CLI interface
│   └── app.py               # Command-line reverse image search
├── streamlit/               # Web UI shared modules
│   ├── core.py              # CLIP model, feature extraction, auto-tuning
│   ├── database.py          # PostgreSQL/pgvector operations
│   ├── streamlit_app.py     # Web interface
│   └── __init__.py
├── ClipVit/                 # Pre-trained CLIP model (ONNX)
│   ├── model.onnx           # CLIP ViT-B/32 encoder
│   ├── preprocessor_config.json
│   └── config.json
├── Images/                  # Image collection directory
│   └── *.jpg, *.jpeg, *.png, *.webp
├── docker-compose.yml       # PostgreSQL database container
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | 3.10+ | Runtime |
| PostgreSQL | 16+ with pgvector | Vector database |
| Podman/Docker | Latest | Container runtime |
| 4GB+ RAM | Required | Model inference |

---

## Installation

### 1. Clone & Navigate

```bash
cd Image
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

Additional dependencies for database:

```bash
pip install psycopg2-binary pgvector
```

### 4. Start PostgreSQL Database

Using Podman:

```bash
podman-compose up -d
```

Or using Docker:

```bash
docker compose up -d
```

Verify container is running:

```bash
podman ps | grep clip_vector_db
# or
docker ps | grep clip_vector_db
```

---

## Usage

### Option 1: Web Interface (Streamlit)

```bash
streamlit run streamlit/streamlit_app.py
```

Open browser at `http://localhost:8501`

**Features:**
- Upload query image
- Auto-detect search strategy (crop/detail vs full product)
- Color palette visualization
- Hybrid ranking with semantic, color, and texture scores
- Top match + similar products display

### Option 2: CLI Interface

```bash
cd python
python app.py
```

Sample output:

```
Connecting to PostgreSQL collection...
Database is empty. Indexing images in 'Images' folder...
Index complete. Total images: 100

--- Reverse Image Search (DB Powered) ---
Enter image path to search (or 'q' to quit): test_image.jpg
Analyzing and searching DB...

Found 3 images with > 80% similarity:
1. similar_product_001.jpg (Score: 0.8532)
2. similar_product_002.jpg (Score: 0.8241)
3. similar_product_003.jpg (Score: 0.8105)
```

---

## Indexing New Images

### Web UI
Click **Update DB Index** in the sidebar. The system will:
1. Scan `Images/` directory
2. Extract global CLIP embeddings
3. Generate region embeddings (3x3 grid, quadrants, center)
4. Compute color distributions (K-means k=5)
5. Compute texture vectors (LBP histogram)
6. Store all features in PostgreSQL

### CLI (Automatic)
On first run, `app.py` automatically indexes all images in the `Images/` folder.

---

## Search Strategies

The system auto-detects query type and adjusts weights:

| Strategy | Detection | Color Weight | Texture Weight | Semantic Weight |
|----------|-----------|--------------|----------------|-----------------|
| **Crop/Detail** | Aspect ratio >1.8 or small area | 5% | 5% | 90% |
| **Color-Dominant** | Uniform colors (low color std) | 25% | 10% | 65% |
| **Pattern-Heavy** | High edge density | 10% | 25% | 65% |
| **Balanced** | Standard product shot | 15% | 10% | 75% |

---

## Database Schema

```sql
-- Main embeddings table
CREATE TABLE image_embeddings (
    id SERIAL PRIMARY KEY,
    filename TEXT UNIQUE NOT NULL,
    embedding VECTOR(512) NOT NULL,
    color_rgb VECTOR(3),
    texture_vector VECTOR(32)
);

-- Region embeddings for localized matching
CREATE TABLE region_embeddings (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    region_type TEXT NOT NULL,
    embedding VECTOR(512) NOT NULL
);

-- Color distribution (top 5 colors per image)
CREATE TABLE color_distribution (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    color VECTOR(3) NOT NULL,
    proportion FLOAT NOT NULL
);
```

---

## Configuration

### Database Connection (database.py:8-14)

```python
db = DatabaseManager(
    host="localhost",
    port=5433,           # Must match docker-compose port
    user="postgres",
    password="root",
    dbname="clip_vector_db"
)
```

### Model Paths (streamlit_app.py:15-17)

```python
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(PARENT_DIR, "Images")
MODEL_DIR = os.path.join(PARENT_DIR, "ClipVit")
```

---

## Troubleshooting

### Database Connection Error

```bash
# Check container status
podman ps

# Restart container
podman-compose restart

# View logs
podman logs clip_vector_db
```

### Out of Memory

- Reduce batch size in `core.py:create_index()`
- Use smaller image resolution
- Process fewer images at once

### No Matches Found

- Ensure images are indexed: `db.get_total_count() > 0`
- Check image formats: jpg, jpeg, png, webp
- Verify embeddings stored: query DB directly

---

## Technologies Used

| Category | Technology |
|----------|------------|
| **AI Model** | CLIP ViT-B/32 (ONNX) |
| **Runtime** | ONNX Runtime |
| **Database** | PostgreSQL 16 + pgvector |
| **Web UI** | Streamlit |
| **Image Processing** | Pillow, NumPy |
| **Color Analysis** | Scikit-learn (K-Means) |
| **Texture** | Local Binary Patterns (custom) |
| **Container** | Podman/Docker Compose |

---

## Performance Notes

- Indexing: ~1-2 seconds per image (with all features)
- Search: <100ms for top-10 results
- Supports 100K+ images with proper indexing
- CLIP embeddings: 512-dimensional vectors
# FastAPI Image Search Service

A production-grade REST API for hybrid reverse image search using FastAPI, PostgreSQL with pgvector, and CLIP embeddings.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                          │
│                     (Uvicorn Server)                       │
└──────────────────────────┬──────────────────────────────────────┘
                          │
         ┌────────────────┼────────────────┐
         ▼                ▼                ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   /api/    │    │  /images/  │    │    /ui/   │
│  (REST)    │    │ (Static)  │    │  (Web UI) │
└─────┬──────┘    └───────────┘    └───────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Services Layer                          │
│  ┌────────────────┐    ┌──────────────────────────┐       │
│  │  CLIPService   │    │   ImageService           │       │
│  │  (ONNX Model)  │    │  (Feature Extraction)    │       │
│  └────────────────┘    └──────────────────────────┘       │
└──────────────────────────┬────────────────────────────────┘
                          │
      ┌───────────────────┴────────────────────┐
      ▼                                        ▼
┌─────────────┐                         ┌─────────────┐
│   CRUD      │                         │  Database   │
│  (SQLAlch) │                         │ (asyncpg)   │
└─────┬──────┘                         └──────┬──────┘
      │                                        │
      ▼                                        ▼
┌─────────────┐                         ┌─────────────┐
│   Models    │                         │  PostgreSQL │
│  (ORM)     │                         │  +pgvector │
└─────────────┘                         └─────────────┘
```

---

## Directory Structure

```
fastapi/
├── app/                        # Main application package
│   ├── __init__.py
│   ├── main.py                  # FastAPI app entry point
│   ├── api/                    # API endpoints
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── index.py       # Index/sync endpoints
│   │           └── search.py    # Search endpoints
│   ├── core/
│   │   └── config.py           # Settings/configuration
│   ├── crud/
│   │   └── image_crud.py      # Database CRUD operations
│   ├── db/
│   │   └── session.py         # SQLAlchemy async session
│   ├── models/
│   │   └── image.py         # ORM models
│   ├── schemas/
│   │   └── image.py         # Pydantic schemas
│   └── services/
│       ├── clip_service.py     # CLIP ONNX model wrapper
│       └── image_service.py # Image processing utilities
├── web/                       # Web UI static files
│   ├── index.html
│   ├── script.js
│   └── style.css
├── .env                       # Environment configuration
├── init_db.py                 # Database initialization script
├── batch_index.py            # Batch indexing script
├── requirements.txt          # Python dependencies
└── README.md               # This file
```

---

## API Endpoints

### Search Endpoints

| Method | Path | Description |
|--------|------|------------|
| POST | `/api/v1/search/` | Hybrid image search using uploaded image |
| GET | `/health` | Health check |

### Index Endpoints

| Method | Path | Description |
|--------|------|------------|
| POST | `/api/v1/index/sync` | Sync images directory with database |

### Static Files

| Path | Description |
|------|------------|
| `/images/{filename}` | Serve indexed images |
| `/ui/` | Web interface |

---

## Request/Response Examples

### Search Request

```bash
curl -X POST "http://localhost:8000/api/v1/search/?limit=6" \
  -F "file=@query_image.jpg"
```

### Search Response

```json
{
  "status": "success",
  "highconfidence": {
    "filename": "matched_product.jpg",
    "total_similarity": 0.9234,
    "semantic_score": 0.89,
    "color_dist_score": 0.78,
    "texture_score": 0.65,
    "confidence_label": "🎯 High Confidence"
  },
  "silimar": [
    {
      "filename": "similar_1.jpg",
      "total_similarity": 0.7123,
      "semantic_score": 0.68,
      "color_dist_score": 0.55,
      "texture_score": 0.48,
      "confidence_label": "🔍 Very Similar"
    }
  ],
  "strategy": "🔍 Crop/Detail Mode",
  "color_weight": 0.05,
  "texture_weight": 0.05,
  "semantic_weight": 0.90
}
```

### Index Response

```json
{
  "status": "completed",
  "processed": 15,
  "total": 120,
  "deleted": 2
}
```

---

## Installation

### 1. Prerequisites

| Component | Version |
|-----------|---------|
| Python | 3.10+ |
| PostgreSQL | 16+ with pgvector |
| Podman/Docker | Latest |

### 2. Install Dependencies

```bash
cd fastapi
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Configure Environment

Edit `.env` file:

```bash
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=root
POSTGRES_SERVER=localhost
POSTGRES_PORT=5433
POSTGRES_DB=clip_search

# Paths
IMAGES_DIR=E:\PrabhatTasks\test\Image\Images
MODEL_DIR=E:\PrabhatTasks\test\Image\ClipVit
```

### 4. Start Database

```bash
# From project root
podman-compose up -d
# or
docker compose up -d
```

### 5. Initialize Database

```bash
cd fastapi
python init_db.py
```

Expected output:

```
Connecting to database at: postgresql+asyncpg://postgres:root@localhost:5433/clip_search
Enabling pgvector extension...
Creating tables...
Database initialized successfully!
```

---

## Usage

### Option 1: Run API Server

```bash
cd fastapi
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server runs at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`

### Option 2: Batch Index Images

```bash
cd fastapi
python batch_index.py
```

### Option 3: Web Interface

The web UI is served at `/ui/` when running the server:

```
http://localhost:8000/ui/
```

---

## Database Schema

### Tables

```sql
-- Main image embeddings
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

-- Color distribution
CREATE TABLE color_distribution (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    color VECTOR(3) NOT NULL,
    proportion FLOAT NOT NULL
);
```

Models are defined in `app/models/image.py`:

```python
class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"
    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    embedding = Column(Vector(512), nullable=False)
    color_rgb = Column(Vector(3))
    texture_vector = Column(Vector(32))

class RegionEmbedding(Base):
    __tablename__ = "region_embeddings"
    ...

class ColorDistribution(Base):
    __tablename__ = "color_distribution"
    ...
```

---

## Services

### CLIPService

Located in `app/services/clip_service.py`:

```python
class CLIPService:
    def __init__(self):
        # Loads ONNX model from settings.MODEL_ONNX_PATH
        
    def get_embedding(self, images, do_center_crop=True):
        # Returns 512-dimensional embeddings
        # Accepts single image or list of images
```

### ImageService

Located in `app/services/image_service.py`:

| Method | Description |
|--------|-------------|
| `auto_tune_weights(image)` | Detect query type and return search weights |
| `extract_foreground(image)` | Remove transparent/white backgrounds |
| `get_color_distribution(image, k=5)` | K-Means color palette extraction |
| `get_texture_vector(image)` | LBP texture histogram (32-dim) |
| `get_image_regions(image)` | Generate region crops (3x3, quadrants, center) |
| `calibrate_confidence(similarity)` | Power-law confidence calibration |
| `get_confidence_label(score)` | Human-readable confidence labels |

---

## Web UI

The web UI provides a browser-based interface for searching:

```
http://localhost:8000/ui/
```

Features:
- Image upload
- Strategy detection display
- Color palette visualization
- Results with confidence scores
- One-click re-indexing

---

## Configuration

### Settings

Located in `app/core/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| PROJECT_NAME | Visual Hybrid Search API | API name |
| VERSION | 1.0.0 | API version |
| API_V1_STR | /api/v1 | API prefix |
| IMAGES_DIR | ../Images | Images directory |
| MODEL_DIR | ../ClipVit | Model directory |
| POSTGRES_USER | postgres | Database user |
| POSTGRES_PASSWORD | root | Database password |
| POSTGRES_SERVER | localhost | Database host |
| POSTGRES_PORT | 5433 | Database port |
| POSTGRES_DB | clip_search | Database name |

---

## Dependencies

```
fastapi==0.111.0
uvicorn==0.30.1
sqlalchemy==2.0.31
asyncpg==0.29.0
pgvector==0.2.5
pillow==10.3.0
numpy==1.26.4
onnxruntime==1.18.0
scikit-learn==1.5.0
python-dotenv==1.0.1
python-multipart==0.0.9
httpx==0.27.0
pydantic-settings==2.3.4
aiofiles==24.1.0
```

---

## Troubleshooting

### Database Connection Error

```bash
# Check container
podman ps | grep clip_vector_db

# Restart container
podman-compose restart

# Check logs
podman logs clip_vector_db
```

### Model Not Found

Ensure `ClipVit/model.onnx` exists and `MODEL_DIR` in `.env` is correct.

### No Search Results

- Run batch indexing: `python batch_index.py`
- Check database: Ensure images are indexed with `SELECT COUNT(*) FROM image_embeddings`

### Import Errors

Ensure you're running from the `fastapi` directory and the project root structure is intact.

---

## Performance

| Operation | Time |
|-----------|------|
| Single embedding generation | ~50ms |
| Full image indexing | ~1-2s |
| Hybrid search (top 10) | <100ms |
| API cold start | ~3-5s |

---

## Integration with Other Modules

This FastAPI service shares core logic with:

- **streamlit/** - Web UI using the same CLIP model and database schema
- **python/** - CLI interface using the same feature extraction

All modules use:
- Same CLIP ONNX model (`ClipVit/model.onnx`)
- Same PostgreSQL database (`clip_vector_db`)
- Same image directory (`Images/`)
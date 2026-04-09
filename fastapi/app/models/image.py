from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from app.db.base import Base

class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, unique=True, nullable=False, index=True)
    embedding = Column(Vector(512), nullable=False)
    color_rgb = Column(Vector(3))

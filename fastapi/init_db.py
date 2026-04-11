import asyncio
import sys
import os

# Add the current directory to sys.path to allow importing from app
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from app.db.session import engine, Base
# Import models to ensure they are registered with Base.metadata
from app.models.image import ImageEmbedding, RegionEmbedding, ColorDistribution

async def init_db():
    print(f"Connecting to database at: {engine.url}")
    try:
        async with engine.begin() as conn:
            print("Enabling pgvector extension...")
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            
            print("Creating tables...")
            # create_all is a synchronous method, so we use run_sync to execute it via the async connection
            await conn.run_sync(Base.metadata.create_all)
        
        print("Database initialized successfully!")
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(init_db())

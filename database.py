import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

class DatabaseManager:
    def __init__(self, host="localhost", port=5433, user="postgres", password="root", dbname="clip_search"):
        self.conn_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "dbname": dbname
        }
        self._init_db()

    def get_connection(self, register=True):
        conn = psycopg2.connect(**self.conn_params)
        if register:
            register_vector(conn)
        return conn

    def _init_db(self):
        """Initializes the database schema."""
        conn_master = psycopg2.connect(
            host=self.conn_params["host"],
            port=self.conn_params["port"],
            user=self.conn_params["user"],
            password=self.conn_params["password"],
            dbname="postgres"
        )
        conn_master.autocommit = True
        cur_master = conn_master.cursor()
        
        cur_master.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (self.conn_params['dbname'],))
        if not cur_master.fetchone():
            cur_master.execute(f'CREATE DATABASE "{self.conn_params["dbname"]}"')
        
        cur_master.close()
        conn_master.close()

        conn = self.get_connection(register=False)
        conn.autocommit = True
        cur = conn.cursor()
        
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Table with both pattern and color vectors
        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                embedding VECTOR(512) NOT NULL,
                color_rgb VECTOR(3)
            )
        """)
        
        # Add column if it doesn't exist (for existing DBs)
        cur.execute("""
            DO $$ 
            BEGIN 
                IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                               WHERE table_name='image_embeddings' AND column_name='color_rgb') THEN
                    ALTER TABLE image_embeddings ADD COLUMN color_rgb VECTOR(3);
                END IF;
            END $$;
        """)
        
        cur.close()
        conn.close()

    def save_embedding(self, filename, embedding, color_rgb=None):
        """Inserts or updates an image embedding and color."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            if color_rgb is not None:
                cur.execute("""
                    INSERT INTO image_embeddings (filename, embedding, color_rgb)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (filename) DO UPDATE 
                    SET embedding = EXCLUDED.embedding,
                        color_rgb = EXCLUDED.color_rgb
                """, (filename, embedding.tolist(), color_rgb.tolist()))
            else:
                cur.execute("""
                    INSERT INTO image_embeddings (filename, embedding)
                    VALUES (%s, %s)
                    ON CONFLICT (filename) DO UPDATE 
                    SET embedding = EXCLUDED.embedding
                """, (filename, embedding.tolist()))
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def search_hybrid(self, query_embedding, query_color, color_weight=0.5, limit=12):
        """
        Performs "Smart Match" hybrid search.
        Uses Pattern Match as a mandatory baseline and Color as a multiplier/refiner.
        """
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            # We use a CTE to first calculate base scores and filter out obvious category mismatches.
            # Then we apply the Color Boost logic.
            cur.execute("""
                WITH BaseMatches AS (
                    SELECT filename, 
                           (1 - (embedding <=> %s::vector)) as pattern_score,
                           (1 - (color_rgb <=> %s::vector)) as color_score
                    FROM image_embeddings
                    WHERE color_rgb IS NOT NULL
                )
                SELECT filename, 
                       -- Smart Formula: Pattern score is the base. Color acts as a multiplier boost.
                       -- Even with weight=1.0, a 0.2 pattern match will stay low.
                       pattern_score * ( (1.0 - %s) + (%s * color_score) ) AS total_similarity,
                       pattern_score,
                       color_score
                FROM BaseMatches
                WHERE pattern_score > 0.45 -- Minimum semantic similarity to be considered "the same type of product"
                ORDER BY total_similarity DESC
                LIMIT %s
            """, (query_embedding, query_color, color_weight, color_weight, limit))
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()

    def get_total_count(self):
        conn = self.get_connection(register=False)
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM image_embeddings")
            return cur.fetchone()[0]
        finally:
            cur.close()
            conn.close()

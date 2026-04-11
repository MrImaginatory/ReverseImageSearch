import psycopg2
from pgvector.psycopg2 import register_vector
import numpy as np

class DatabaseManager:
    def __init__(self, host="localhost", port=5433, user="postgres", password="root", dbname="clip_vector_db"):
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
        
        # Table with pattern, color, and texture vectors
        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                embedding VECTOR(512) NOT NULL,
                color_rgb VECTOR(3),
                texture_vector VECTOR(32)
            )
        """)
        
        # Add columns if they don't exist (for existing DBs)
        for col, size in [('color_rgb', 3), ('texture_vector', 32)]:
            cur.execute(f"""
                DO $$ 
                BEGIN 
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                                   WHERE table_name='image_embeddings' AND column_name='{col}') THEN
                        ALTER TABLE image_embeddings ADD COLUMN {col} VECTOR({size});
                    END IF;
                END $$;
            """)
        
        cur.close()
        conn.close()

    def save_embedding(self, filename, embedding, color_rgb=None, texture_vec=None):
        """Inserts or updates an image embedding, color, and texture."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("""
                INSERT INTO image_embeddings (filename, embedding, color_rgb, texture_vector)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (filename) DO UPDATE 
                SET embedding = EXCLUDED.embedding,
                    color_rgb = EXCLUDED.color_rgb,
                    texture_vector = EXCLUDED.texture_vector
            """, (
                filename, 
                embedding.tolist(), 
                color_rgb.tolist() if color_rgb is not None else None,
                texture_vec.tolist() if texture_vec is not None else None
            ))
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def delete_embedding(self, filename):
        """Removes an image from the database."""
        conn = self.get_connection(register=False)
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM image_embeddings WHERE filename = %s", (filename,))
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def get_all_filenames(self):
        """Returns a set of all indexed filenames."""
        conn = self.get_connection(register=False)
        cur = conn.cursor()
        try:
            cur.execute("SELECT filename FROM image_embeddings")
            return set(row[0] for row in cur.fetchall())
        finally:
            cur.close()
            conn.close()

    def search_hybrid(self, query_embedding, query_color, query_texture=None, color_weight=0.3, texture_weight=0.2, limit=12):
        """
        Performs "Smart Match" hybrid search using Pattern, Color, and Texture.
        """
        conn = self.get_connection()
        cur = conn.cursor()
        
        # Ensure weights don't exceed 1.0 (Pattern baseline is at least 0.1)
        # However, we'll handle the logic such that Pattern is the scale factor.
        
        try:
            cur.execute("""
                WITH BaseMatches AS (
                    SELECT filename, 
                           (1 - (embedding <=> %s::vector)) as semantic_score,
                           (1 - (color_rgb <=> %s::vector)) as color_score,
                           (1 - (texture_vector <=> %s::vector)) as texture_score
                    FROM image_embeddings
                    WHERE color_rgb IS NOT NULL AND texture_vector IS NOT NULL
                )
                SELECT filename, 
                       semantic_score * ( (1.0 - %s - %s) + (%s * color_score) + (%s * texture_score) ) AS total_similarity,
                       semantic_score,
                       color_score,
                       texture_score
                FROM BaseMatches
                WHERE semantic_score > 0.40 
                ORDER BY total_similarity DESC
                LIMIT %s
            """, (query_embedding, query_color, query_texture.tolist() if query_texture is not None else query_color, 
                  color_weight, texture_weight, color_weight, texture_weight, limit))
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

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
        self.table_name = "googlevit_embeddings"
        self._init_db()

    def get_connection(self, register=True):
        conn = psycopg2.connect(**self.conn_params)
        if register:
            register_vector(conn)
        return conn

    def _init_db(self):
        """Initializes the database schema for Google ViT embeddings."""
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
        
        # Table with Google ViT (768), color (3), and texture (32) vectors
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                embedding VECTOR(768) NOT NULL,
                color_rgb VECTOR(3),
                texture_vector VECTOR(32)
            )
        """)
        
        cur.close()
        conn.close()

    def save_embedding(self, filename, embedding, color_rgb=None, texture_vec=None):
        """Inserts or updates a Google ViT embedding, color, and texture."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute(f"""
                INSERT INTO {self.table_name} (filename, embedding, color_rgb, texture_vector)
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
            cur.execute(f"DELETE FROM {self.table_name} WHERE filename = %s", (filename,))
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def get_all_filenames(self):
        """Returns a set of all indexed filenames."""
        conn = self.get_connection(register=False)
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT filename FROM {self.table_name}")
            return set(row[0] for row in cur.fetchall())
        finally:
            cur.close()
            conn.close()

    def search_hybrid(self, query_embedding, query_color, query_texture=None, color_weight=0.3, texture_weight=0.2, limit=12):
        """
        Performs "Smart Match" hybrid search using Pattern (ViT), Color, and Texture.
        """
        conn = self.get_connection()
        cur = conn.cursor()
        
        try:
            cur.execute(f"""
                WITH BaseMatches AS (
                    SELECT filename, 
                           (1 - (embedding <=> %s::vector)) as semantic_score,
                           (1 - (color_rgb <=> %s::vector)) as color_score,
                           (1 - (texture_vector <=> %s::vector)) as texture_score
                    FROM {self.table_name}
                    WHERE color_rgb IS NOT NULL AND texture_vector IS NOT NULL
                )
                SELECT filename, 
                       semantic_score * ( (1.0 - %s - %s) + (%s * color_score) + (%s * texture_score) ) AS total_similarity,
                       semantic_score,
                       color_score,
                       texture_score
                FROM BaseMatches
                WHERE semantic_score > 0.30 
                ORDER BY total_similarity DESC
                LIMIT %s
            """, (query_embedding.tolist(), query_color.tolist(), query_texture.tolist() if query_texture is not None else query_color.tolist(), 
                  color_weight, texture_weight, color_weight, texture_weight, limit))
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()

    def get_total_count(self):
        conn = self.get_connection(register=False)
        cur = conn.cursor()
        try:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name}")
            return cur.fetchone()[0]
        finally:
            cur.close()
            conn.close()

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
        # Connect to default postgres DB to manage other databases
        conn_master = psycopg2.connect(
            host=self.conn_params["host"],
            port=self.conn_params["port"],
            user=self.conn_params["user"],
            password=self.conn_params["password"],
            dbname="postgres"
        )
        conn_master.autocommit = True
        cur_master = conn_master.cursor()
        
        # Check if target database exists
        cur_master.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = %s", (self.conn_params['dbname'],))
        if not cur_master.fetchone():
            cur_master.execute(f'CREATE DATABASE "{self.conn_params["dbname"]}"')
        
        cur_master.close()
        conn_master.close()

        # Connect to the target DB to create extensions and tables
        # Use register=False because the extension might not exist yet!
        conn = self.get_connection(register=False)
        conn.autocommit = True
        cur = conn.cursor()
        
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Now we can register vector on the original connection if we wanted, 
        # but for internal setup we just need to create the table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                embedding VECTOR(512) NOT NULL
            )
        """)
        
        cur.close()
        conn.close()

    def save_embedding(self, filename, embedding):
        """Inserts or updates an image embedding."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
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

    def search_similarity(self, query_embedding, limit=10):
        """Performs vector similarity search."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # Using <-> for L2 distance or <=> for cosine distance
            # Since CLIP embeddings are normalized, they are equivalent
            cur.execute("""
                SELECT filename, 1 - (embedding <=> %s::vector) AS cosine_similarity
                FROM image_embeddings
                ORDER BY cosine_similarity DESC
                LIMIT %s
            """, (query_embedding, limit))
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()

    def get_total_count(self):
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT COUNT(*) FROM image_embeddings")
            return cur.fetchone()[0]
        finally:
            cur.close()
            conn.close()

    def get_all_data(self):
        """Returns all embeddings and filenames (fallback for legacy logic if needed)."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            cur.execute("SELECT embedding, filename FROM image_embeddings")
            rows = cur.fetchall()
            if not rows:
                return np.array([]), []
            embeddings = np.array([r[0] for r in rows])
            filenames = [r[1] for r in rows]
            return embeddings, filenames
        finally:
            cur.close()
            conn.close()

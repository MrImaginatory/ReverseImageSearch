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
                         EXECUTE 'ALTER TABLE image_embeddings ADD COLUMN ' || quote_ident('{col}') || ' VECTOR({size})';
                    END IF;
                END $$;
            """)
            
        # [NEW] Region Embeddings Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS region_embeddings (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                region_type TEXT NOT NULL,
                embedding VECTOR(512) NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_region_filename ON region_embeddings(filename)")
        
        # [NEW] Color Distribution Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS color_distribution (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                color VECTOR(3) NOT NULL,
                proportion FLOAT NOT NULL
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_color_filename ON color_distribution(filename)")
        
        cur.close()
        conn.close()

    def save_embedding(self, filename, embedding, color_rgb=None, texture_vec=None, regions=None, color_dist=None):
        """Inserts or updates an image embedding, color palette, and regions."""
        conn = self.get_connection()
        cur = conn.cursor()
        try:
            # 1. Main Embedding
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
            
            # 2. Add regions if provided
            if regions:
                cur.execute("DELETE FROM region_embeddings WHERE filename = %s", (filename,))
                for r_type, r_emb in regions:
                    cur.execute("INSERT INTO region_embeddings (filename, region_type, embedding) VALUES (%s, %s, %s)",
                                (filename, r_type, r_emb.tolist()))
                                
            # 3. Add color distribution if provided
            if color_dist:
                cur.execute("DELETE FROM color_distribution WHERE filename = %s", (filename,))
                for color, prop in color_dist:
                    cur.execute("INSERT INTO color_distribution (filename, color, proportion) VALUES (%s, %s, %s)",
                                (filename, color.tolist(), prop))
                                
            conn.commit()
        finally:
            cur.close()
            conn.close()

    def delete_embedding(self, filename):
        """Removes an image and its regions from the database."""
        conn = self.get_connection(register=False)
        cur = conn.cursor()
        try:
            cur.execute("DELETE FROM image_embeddings WHERE filename = %s", (filename,))
            cur.execute("DELETE FROM region_embeddings WHERE filename = %s", (filename,))
            cur.execute("DELETE FROM color_distribution WHERE filename = %s", (filename,))
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

    def search_hybrid(self, query_embedding, query_color_dist, query_texture=None, color_weight=0.3, texture_weight=0.2, limit=12):
        """
        Performs Advanced Hybrid Search using Localized Regions and Color Distributions.
        """
        conn = self.get_connection()
        cur = conn.cursor()
        
        # For query color distribution, we'll simplify and use a weighted sum of the top 3 query colors
        # vs the best matching colors in DB images.
        
        try:
            # 1. Fetch scores for Global and Localized Pattern Matching
            # We use a CTE to find the best matching region for each image
            cur.execute("""
                WITH LocalizedScores AS (
                    SELECT filename, MAX(1 - (embedding <=> %s::vector)) as best_region_score
                    FROM region_embeddings
                    GROUP BY filename
                ),
                ColorDistributionScores AS (
                    SELECT filename, SUM((1 - (color <=> %s::vector)) * proportion) as color_dist_score
                    FROM color_distribution
                    GROUP BY filename
                ),
                BaseMatches AS (
                    SELECT e.filename, 
                           (1 - (e.embedding <=> %s::vector)) as global_semantic_score,
                           COALESCE(ls.best_region_score, 0) as local_semantic_score,
                           COALESCE(cs.color_dist_score, 0) as color_dist_score,
                           (1 - (e.texture_vector <=> %s::vector)) as texture_score
                    FROM image_embeddings e
                    LEFT JOIN LocalizedScores ls ON e.filename = ls.filename
                    LEFT JOIN ColorDistributionScores cs ON e.filename = cs.filename
                )
                SELECT filename, 
                       ( (1.0 - %s - %s) * GREATEST(global_semantic_score, local_semantic_score) ) + 
                       (%s * color_dist_score) + 
                       (%s * texture_score) AS total_similarity,
                       GREATEST(global_semantic_score, local_semantic_score) as semantic_score,
                       color_dist_score,
                       texture_score
                FROM BaseMatches
                WHERE global_semantic_score > 0.35 OR local_semantic_score > 0.45
                ORDER BY total_similarity DESC
                LIMIT %s
            """, (
                query_embedding, 
                query_color_dist[0][0].tolist(), # Using top query color for simplicity in SQL
                query_embedding, 
                query_texture.tolist() if query_texture is not None else query_embedding, 
                color_weight, texture_weight, color_weight, texture_weight, limit
            ))
            return cur.fetchall()
        finally:
            cur.close()
            conn.close()

    def get_incomplete_filenames(self):
        """Returns filenames that are missing advanced features (regions)."""
        conn = self.get_connection(register=False)
        cur = conn.cursor()
        try:
            cur.execute("""
                SELECT e.filename FROM image_embeddings e
                LEFT JOIN region_embeddings r ON e.filename = r.filename
                WHERE r.filename IS NULL
            """)
            return set(row[0] for row in cur.fetchall())
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

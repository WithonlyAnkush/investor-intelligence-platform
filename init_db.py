from sqlalchemy import text
from database import engine, Base
import models  # Ensures models are registered with Base metadata

def initialize_database():
    print("Connecting to Neon PostgreSQL...")
    with engine.begin() as conn:
        # Enable pgvector extension inside the database
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        print("Vector extension verified.")
    
    # Create tables if they do not already exist
    Base.metadata.create_all(bind=engine)
    print("Tables (documents, document_chunks, extracted_kpis) created successfully.")

if __name__ == "__main__":
    initialize_database()
from sqlalchemy import Column, Integer, String, Text, Numeric, ForeignKey, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    company_name = Column(String(255), nullable=False)
    fiscal_year = Column(Integer, nullable=False)
    filename = Column(String(255), nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("DocumentChunk", back_populates="document", cascade="all, delete-orphan")
    kpis = relationship("ExtractedKPI", back_populates="document", cascade="all, delete-orphan")


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    
    # 384 dimensions matches the all-MiniLM-L6-v2 embedding model
    embedding = Column(Vector(384), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")


class ExtractedKPI(Base):
    __tablename__ = "extracted_kpis"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    kpi_name = Column(String(100), nullable=False)
    kpi_value = Column(Numeric(15, 2), nullable=True)
    currency = Column(String(10), default="USD")
    fiscal_year = Column(Integer, nullable=False)
    source_context = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="kpis")
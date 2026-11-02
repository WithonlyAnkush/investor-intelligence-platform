import os
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import select

from database import get_db
from models import Document, DocumentChunk, ExtractedKPI
import ai_services

# Initialize the FastAPI server
app = FastAPI(
    title="Investor Intelligence Platform",
    description="AI Backend for Document Processing and RAG",
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Investor Intelligence Platform API is running."}

@app.post("/api/v1/upload-report")
async def upload_report(
    company_name: str = Form(...),
    fiscal_year: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # 1. Save the uploaded file temporarily to the local disk
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
        
    try:
        # 2. Read PDF text and slice it into overlapping chunks
        raw_text = ai_services.extract_text_from_pdf(temp_path)
        if not raw_text:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
            
        chunks = ai_services.chunk_text(raw_text)
        
        # 3. Create the parent Document record in the database
        new_doc = Document(company_name=company_name, fiscal_year=fiscal_year, filename=file.filename)
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        
        # 4. Embed chunks via Hugging Face and store them with vectors
        for i, text in enumerate(chunks):
            vector = ai_services.get_embedding(text)
            chunk_record = DocumentChunk(
                document_id=new_doc.id,
                chunk_index=i,
                chunk_text=text,
                embedding=vector
            )
            db.add(chunk_record)
            
        # 5. Extract KPIs via Groq and store the structured JSON
        kpis = ai_services.extract_kpis_with_llm(raw_text, fiscal_year)
        for kpi in kpis:
            try:
                raw_val = kpi.get("kpi_value")
                clean_val = float(raw_val) if raw_val is not None else None
            except (ValueError, TypeError):
                clean_val = None
                
            kpi_record = ExtractedKPI(
                document_id=new_doc.id,
                kpi_name=str(kpi.get("kpi_name", "Unknown"))[:100],
                kpi_value=clean_val,
                currency=str(kpi.get("currency", "USD"))[:10],
                fiscal_year=fiscal_year,
                source_context=str(kpi.get("source_context", ""))
            )
            db.add(kpi_record)
            
        db.commit()
        
    finally:
        # 6. Clean up: Delete the temporary PDF file from your computer
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return {
        "message": "Report processed successfully", 
        "document_id": new_doc.id, 
        "kpis_extracted": len(kpis)
    }

@app.post("/api/v1/chat")
async def chat_with_report(query: str, db: Session = Depends(get_db)):
    # 1. Convert the user's question into a 384-dimensional vector
    query_vector = ai_services.get_embedding(query)
    
    # 2. Semantic Search: Find the 4 closest text chunks using pgvector's cosine distance
    results = db.scalars(
        select(DocumentChunk)
        .order_by(DocumentChunk.embedding.cosine_distance(query_vector))
        .limit(4)
    ).all()
    
    if not results:
        return {"query": query, "answer": "No relevant documents found in the database."}
        
    # 3. Combine the retrieved chunk texts to form a contextual prompt
    contexts = [chunk.chunk_text for chunk in results]
    
    # 4. Pass the context and query to Groq for the final answer
    answer = ai_services.generate_rag_answer(query, contexts)
    
    return {
        "query": query, 
        "answer": answer,
        "sources_used": [chunk.chunk_text[:150] + "..." for chunk in results]
    }

@app.get("/api/v1/kpis/{document_id}")
def get_kpis_for_document(document_id: int, db: Session = Depends(get_db)):
    """Fetch all structured financial metrics extracted for a specific document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
        
    kpis = db.query(ExtractedKPI).filter(ExtractedKPI.document_id == document_id).all()
    
    return {
        "company_name": doc.company_name,
        "fiscal_year": doc.fiscal_year,
        "filename": doc.filename,
        "kpis": [
            {
                "kpi_name": kpi.kpi_name,
                "kpi_value": float(kpi.kpi_value) if kpi.kpi_value is not None else None,
                "currency": kpi.currency,
                "source_context": kpi.source_context
            }
            for kpi in kpis
        ]
    }
# 📈 AI-Powered Investor Intelligence Platform

An end-to-end Retrieval-Augmented Generation (RAG) and structured financial intelligence platform designed to ingest corporate annual reports, execute high-speed semantic vector searches, extract structured financial KPIs via Large Language Models, and present insights through an interactive dashboard and chat interface.

---

## 🚀 Live Demo & Architecture
* **Backend API (FastAPI):** Hosted live on [Render](https://render.com) (`https://investor-intelligence-backend.onrender.com`)
* **Frontend UI (Streamlit):** Hosted live on Streamlit Community Cloud

---

## 🛠️ Tech Stack & Architecture

* **Backend & API Framework:** Python, FastAPI, Uvicorn
* **Frontend Interface:** Streamlit (Multi-page reactive UI)
* **Database & Vector Store:** Neon PostgreSQL with the `pgvector` extension for high-performance cosine similarity search.
* **Embeddings & NLP:** `fastembed` (running `sentence-transformers/all-MiniLM-L6-v2` via ONNX Runtime for a lightweight, memory-efficient deployment footprint).
* **LLM & Extraction Engine:** Groq API (leveraging Llama 3 for low-latency inference and structured JSON extraction).
* **Document Processing:** `pypdf` for text parsing and custom sliding-window semantic chunking.

---

## 🌟 Key Features

1. **Automated PDF Ingestion & Chunking:** Parses unstructured financial reports, strips noise, and slices text into overlapping character windows to preserve semantic context.
2. **Local Vector Embeddings (`fastembed`):** Generates 384-dimensional dense vectors locally with a minimal memory footprint (<150 MB), bypassing heavy PyTorch dependencies for free-tier cloud deployment.
3. **Semantic RAG Chatbot:** Performs vector similarity search via PostgreSQL `pgvector` (`cosine_distance`) to fetch relevant document chunks and grounds LLM responses to prevent hallucinations.
4. **AI-Driven KPI Extraction:** Automatically extracts key financial metrics (Revenue, Net Income, Gross Margin, Operating Income) into relational tables for instant dashboard queries.
5. **Production-Ready & Cost-Optimized:** Fully deployed on free-tier cloud infrastructure (Render & Streamlit Cloud) with optimized cold-start handling.

---

## ⚙️ Local Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/WithonlyAnkush/investor-intelligence-platform.git](https://github.com/WithonlyAnkush/investor-intelligence-platform.git)
cd investor-intelligence-platform
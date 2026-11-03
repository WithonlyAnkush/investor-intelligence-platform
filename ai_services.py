import os
import json
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv
from fastembed import TextEmbedding

load_dotenv()

# Initialize lightweight embedding model (uses ONNX, tiny memory footprint)
print("Loading embedding model...")
embedding_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Groq client
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing from your .env file.")

groq_client = Groq(api_key=GROQ_API_KEY)


def extract_text_from_pdf(file_path: str) -> str:
    """Reads all readable pages from a PDF file and returns cleaned text."""
    reader = PdfReader(file_path)
    extracted_pages = []
    
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_pages.append(page_text.strip())
            
    return "\n\n".join(extracted_pages)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """
    Slices text into overlapping character windows to ensure context 
    is not cut abruptly across boundaries.
    """
    chunks = []
    start = 0
    total_len = len(text)

    while start < total_len:
        end = min(start + chunk_size, total_len)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks


def get_embedding(text: str) -> list[float]:
    """
    Generates a 384-dimensional vector embedding using fastembed.
    """
    # fastembed returns a generator of numpy arrays, we convert the first result to a list
    embeddings_generator = embedding_model.embed([text])
    vector = next(embeddings_generator)
    return vector.tolist()


def extract_kpis_with_llm(report_summary: str, fiscal_year: int) -> list[dict]:
    """
    Instructs Groq's Llama 3 model to find financial metrics (Revenue, Operating Margin, etc.)
    and return strictly formatted JSON.
    """
    system_prompt = (
        "You are an expert financial analyst. Extract key financial metrics (such as Total Revenue, "
        "Net Income, Operating Income, Gross Margin, Diluted EPS) from the provided annual report text. "
        "Return ONLY a valid JSON array of objects. Do not wrap in markdown quotes if possible, "
        "and do not include commentary. Each object must contain:\n"
        "- 'kpi_name': name of the metric (string)\n"
        "- 'kpi_value': purely numerical value (float/integer, no commas or currency symbols)\n"
        "- 'currency': ISO currency code like 'USD', 'EUR', or 'INR' (string)\n"
        "- 'source_context': short excerpt showing where the number was found (string)"
    )

    # Limit snippet size to fit comfortably within prompt window
    user_content = f"Fiscal Year: {fiscal_year}\n\nDocument Excerpt:\n{report_summary[:6000]}"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1
    )

    content = response.choices[0].message.content.strip()
    
    # Strip potential markdown formatting if returned by the LLM
    if content.startswith("```json"):
        content = content[7:]
    if content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]

    try:
        data = json.loads(content.strip())
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        print(f"Warning: Failed to parse LLM response as JSON: {content}")
        return []


def generate_rag_answer(query: str, retrieved_contexts: list[str]) -> str:
    """Combines retrieved database chunks with the user's question to generate a grounded answer."""
    context_block = "\n---\n".join(retrieved_contexts)
    
    system_prompt = (
        "You are a financial advisor answering questions about an annual report. "
        "Answer the user's question using ONLY the retrieved context excerpts provided below. "
        "If the answer cannot be found in the context, explicitly state that the information "
        "is not present in the document."
    )

    user_content = f"Retrieved Report Sections:\n{context_block}\n\nUser Question: {query}"

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content
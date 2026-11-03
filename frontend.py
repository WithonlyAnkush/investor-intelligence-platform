import streamlit as st
import requests

# FastAPI Backend URL
API_URL = "https://investor-intelligence-backend.onrender.com"

st.set_page_config(
    page_title="Investor Intelligence Platform", 
    page_icon="📈", 
    layout="wide"
)

st.title("📊 AI-Powered Investor Intelligence Platform")
st.markdown("Upload annual financial reports, extract structured KPIs via AI, and query reports using semantic search.")

# Sidebar Navigation
page = st.sidebar.selectbox("Navigation", ["Upload Report", "RAG Chatbot", "KPI Dashboard"])

# ==================== PAGE 1: UPLOAD REPORT ====================
if page == "Upload Report":
    st.header("📄 Upload Annual Report PDF")
    
    col1, col2 = st.columns(2)
    with col1:
        company_name = st.text_input("Company Name", "Apple")
    with col2:
        fiscal_year = st.number_input("Fiscal Year", min_value=2000, max_value=2100, value=2024)
        
    uploaded_file = st.file_uploader("Choose an annual report PDF file", type="pdf")

    if st.button("Process & Embed Document", type="primary"):
        if uploaded_file is not None:
            with st.spinner("Processing PDF, generating local vectors, storing in PostgreSQL, and extracting KPIs..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"company_name": company_name, "fiscal_year": fiscal_year}
                
                try:
                    response = requests.post(f"{API_URL}/api/v1/upload-report", data=data, files=files)
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        st.success("Report processed and embedded successfully!")
                        st.info(f"**Document ID:** {res_json.get('document_id')} | **KPIs Extracted:** {res_json.get('kpis_extracted')}")
                    else:
                        st.error(f"Server Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend. Make sure your FastAPI server is running!")
        else:
                        st.warning("Please upload a PDF file first.")

# ==================== PAGE 2: RAG CHATBOT ====================
elif page == "RAG Chatbot":
    st.header("💬 Investor Intelligence Chatbot")
    st.markdown("Ask financial questions. The system will retrieve relevant document chunks using `pgvector` and formulate an answer using AI.")
    
    query = st.text_input("Enter your financial question:", placeholder="e.g., What was the total revenue and net income for the year?")
    
    if st.button("Ask AI", type="primary"):
        if query:
            with st.spinner("Searching vector database and synthesizing answer..."):
                try:
                    response = requests.post(f"{API_URL}/api/v1/chat", params={"query": query})
                    
                    if response.status_code == 200:
                        res_json = response.json()
                        st.markdown("### Answer")
                        st.write(res_json.get("answer"))
                        
                        with st.expander("🔍 View Retrieved Source Excerpts"):
                            for idx, source in enumerate(res_json.get("sources_used", [])):
                                st.markdown(f"**Chunk {idx+1}:**")
                                st.text(source)
                    else:
                        st.error(f"Error: {response.text}")
                except requests.exceptions.ConnectionError:
                    st.error("Could not connect to FastAPI backend.")
        else:
            st.warning("Please enter a question.")

# ==================== PAGE 3: KPI DASHBOARD ====================
elif page == "KPI Dashboard":
    st.header("📈 Financial KPI Dashboard")
    st.markdown("View structured financial metrics extracted automatically from uploaded reports.")
    
    doc_id = st.number_input("Enter Document ID", min_value=1, value=1, step=1)
    
    if st.button("Fetch Dashboard Metrics", type="primary"):
        with st.spinner("Fetching relational KPI records from database..."):
            try:
                response = requests.get(f"{API_URL}/api/v1/kpis/{doc_id}")
                
                if response.status_code == 200:
                    data = response.json()
                    st.subheader(f"Company: {data.get('company_name')} ({data.get('fiscal_year')})")
                    st.caption(f"Source File: {data.get('filename')}")
                    
                    kpis = data.get("kpis", [])
                    if kpis:
                        st.table(kpis)
                    else:
                        st.info("No structured KPIs found for this document ID.")
                else:
                    st.error("Document ID not found in database.")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to FastAPI backend.")